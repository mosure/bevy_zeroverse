//! Qualification deadlines follow the actual adapter, without changing render quality.
use std::time::Duration;

pub fn for_adapter(adapter: Option<wgpu::DeviceType>, hardware_seconds: u64) -> Duration {
    let multiplier = if adapter == Some(wgpu::DeviceType::Cpu) {
        10
    } else {
        1
    };
    Duration::from_secs(hardware_seconds.saturating_mul(multiplier))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_software_adapters_extend_the_bounded_qualification_budget() {
        for adapter in [
            None,
            Some(wgpu::DeviceType::DiscreteGpu),
            Some(wgpu::DeviceType::IntegratedGpu),
            Some(wgpu::DeviceType::VirtualGpu),
            Some(wgpu::DeviceType::Other),
        ] {
            assert_eq!(for_adapter(adapter, 60), Duration::from_secs(60));
        }
        assert_eq!(
            for_adapter(Some(wgpu::DeviceType::Cpu), 60),
            Duration::from_secs(600)
        );
        assert_eq!(
            for_adapter(Some(wgpu::DeviceType::Cpu), 180),
            Duration::from_secs(1800)
        );
    }
}
