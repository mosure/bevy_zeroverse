# bevy_zeroverse_capture

Lightweight shared contracts for the generator and its publication tools:
generator/schema identity, six-channel publication protocol and deterministic
co-visibility camera colors. The optional `provenance` feature hashes explicit
renderer build inputs, including Cargo.lock and local WGPU patches.
Text build inputs use canonical LF line endings, so CRLF checkouts produce the
same identity. Binary inputs and rendered artifact hashes remain byte-exact.

It has no Bevy, GPU, model, browser or publication/typesetting initialization.
