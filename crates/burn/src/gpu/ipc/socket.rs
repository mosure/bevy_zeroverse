//! Length-framed Unix stream with SCM_RIGHTS. FDs are owned, never JSON integers.
use super::*;
use std::{
    io::{Read, Write},
    mem::size_of,
    os::fd::{AsRawFd, FromRawFd, OwnedFd},
    os::unix::net::UnixStream,
};

pub(super) struct Socket(pub UnixStream);
impl Socket {
    pub fn authenticate(&self) -> Result<u32> {
        let mut peer: libc::ucred = unsafe { std::mem::zeroed() };
        let mut size = size_of::<libc::ucred>() as libc::socklen_t;
        let result = unsafe {
            libc::getsockopt(
                self.0.as_raw_fd(),
                libc::SOL_SOCKET,
                libc::SO_PEERCRED,
                (&mut peer as *mut libc::ucred).cast(),
                &mut size,
            )
        };
        ensure!(
            result == 0,
            "cannot authenticate Unix peer: {}",
            std::io::Error::last_os_error()
        );
        ensure!(
            peer.uid == unsafe { libc::geteuid() },
            "CUDA IPC peer has different uid"
        );
        Ok(peer.pid as u32)
    }
    pub fn send(&mut self, message: &Message, fds: &[OwnedFd]) -> Result<()> {
        ensure!(fds.len() <= 2, "too many ancillary FDs");
        let body = serde_json::to_vec(message)?;
        ensure!(
            body.len() <= wire::MAX_MESSAGE_BYTES,
            "IPC message too large"
        );
        let mut header = [0u8; 8];
        header[..4].copy_from_slice(&(body.len() as u32).to_le_bytes());
        header[4..].copy_from_slice(&(fds.len() as u32).to_le_bytes());
        let mut iov = libc::iovec {
            iov_base: header.as_mut_ptr().cast(),
            iov_len: header.len(),
        };
        // usize storage gives ancillary cmsghdr its required alignment.
        let mut ancillary = [0usize; 16];
        let mut msg: libc::msghdr = unsafe { std::mem::zeroed() };
        msg.msg_iov = &mut iov;
        msg.msg_iovlen = 1;
        if !fds.is_empty() {
            msg.msg_control = ancillary.as_mut_ptr().cast();
            msg.msg_controllen =
                unsafe { libc::CMSG_SPACE((fds.len() * size_of::<libc::c_int>()) as u32) } as usize;
            unsafe {
                let cmsg = libc::CMSG_FIRSTHDR(&msg);
                (*cmsg).cmsg_level = libc::SOL_SOCKET;
                (*cmsg).cmsg_type = libc::SCM_RIGHTS;
                (*cmsg).cmsg_len =
                    libc::CMSG_LEN((fds.len() * size_of::<libc::c_int>()) as u32) as usize;
                for (i, fd) in fds.iter().enumerate() {
                    libc::CMSG_DATA(cmsg)
                        .cast::<libc::c_int>()
                        .add(i)
                        .write(fd.as_raw_fd());
                }
            }
        }
        let sent = loop {
            let n = unsafe { libc::sendmsg(self.0.as_raw_fd(), &msg, libc::MSG_NOSIGNAL) };
            if n < 0 && std::io::Error::last_os_error().kind() == std::io::ErrorKind::Interrupted {
                continue;
            }
            break n;
        };
        ensure!(
            sent > 0,
            "IPC header send failed: {}",
            std::io::Error::last_os_error()
        );
        self.0.write_all(&header[sent as usize..])?;
        self.0.write_all(&body)?;
        Ok(())
    }
    pub fn receive(&mut self) -> Result<(Message, Vec<OwnedFd>)> {
        let mut header = [0u8; 8];
        let mut ancillary = [0usize; 16];
        let mut iov = libc::iovec {
            iov_base: header.as_mut_ptr().cast(),
            iov_len: 8,
        };
        let mut msg: libc::msghdr = unsafe { std::mem::zeroed() };
        msg.msg_iov = &mut iov;
        msg.msg_iovlen = 1;
        msg.msg_control = ancillary.as_mut_ptr().cast();
        msg.msg_controllen = std::mem::size_of_val(&ancillary);
        let received = loop {
            let n = unsafe { libc::recvmsg(self.0.as_raw_fd(), &mut msg, libc::MSG_CMSG_CLOEXEC) };
            if n < 0 && std::io::Error::last_os_error().kind() == std::io::ErrorKind::Interrupted {
                continue;
            }
            break n;
        };
        ensure!(
            received > 0,
            "IPC disconnected/read failed: {}",
            std::io::Error::last_os_error()
        );
        let mut fds = Vec::new();
        let mut valid = true;
        unsafe {
            let mut cmsg = libc::CMSG_FIRSTHDR(&msg);
            while !cmsg.is_null() {
                if (*cmsg).cmsg_level == libc::SOL_SOCKET && (*cmsg).cmsg_type == libc::SCM_RIGHTS {
                    let count =
                        ((*cmsg).cmsg_len - libc::CMSG_LEN(0) as usize) / size_of::<libc::c_int>();
                    for i in 0..count {
                        fds.push(OwnedFd::from_raw_fd(
                            libc::CMSG_DATA(cmsg).cast::<libc::c_int>().add(i).read(),
                        ));
                    }
                } else {
                    valid = false;
                }
                cmsg = libc::CMSG_NXTHDR(&msg, cmsg);
            }
        }
        // All received descriptors have RAII ownership even on malformed frames.
        ensure!(
            valid && msg.msg_flags & (libc::MSG_CTRUNC | libc::MSG_TRUNC) == 0 && fds.len() <= 2,
            "truncated/unexpected IPC ancillary data"
        );
        self.0.read_exact(&mut header[received as usize..])?;
        let bytes = u32::from_le_bytes(header[..4].try_into().unwrap()) as usize;
        let expected = u32::from_le_bytes(header[4..].try_into().unwrap()) as usize;
        ensure!(
            bytes > 0 && bytes <= wire::MAX_MESSAGE_BYTES && expected == fds.len(),
            "invalid IPC frame bounds/FD count"
        );
        let mut body = vec![0; bytes];
        self.0.read_exact(&mut body)?;
        Ok((serde_json::from_slice(&body)?, fds))
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn ancillary_fds_survive_sender_close_and_framing_is_exact() {
        let (a, b) = UnixStream::pair().unwrap();
        let mut a = Socket(a);
        let mut b = Socket(b);
        assert_eq!(a.authenticate().unwrap(), std::process::id());
        let fd: OwnedFd = std::fs::File::open("/dev/null").unwrap().into();
        a.send(&Message::Stop, &[fd]).unwrap();
        a.send(&Message::Stop, &[]).unwrap();
        let (m, fds) = b.receive().unwrap();
        assert!(matches!(m, Message::Stop));
        assert_eq!(fds.len(), 1);
        assert!(unsafe { libc::fcntl(fds[0].as_raw_fd(), libc::F_GETFD) } & libc::FD_CLOEXEC != 0);
        assert!(b.receive().unwrap().1.is_empty());
    }
    #[test]
    fn oversized_frame_is_rejected_before_allocation() {
        let (mut a, b) = UnixStream::pair().unwrap();
        a.write_all(&[255, 255, 255, 127, 0, 0, 0, 0]).unwrap();
        assert!(Socket(b).receive().is_err());
    }
}
