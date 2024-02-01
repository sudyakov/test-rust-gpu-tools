use rust_gpu_tools::{cuda, opencl, program_closures, Device, GPUError, Program, Vendor};
// Получаем первое доступное CUDA устройство из принимаемого списка.
fn get_cuda_device(devices: Vec<&Device>) -> Result<&Device, GPUError> {
    for device in devices {
        if device.vendor() == Vendor::Nvidia {
            return Ok(device);
        }
    }
    Err(GPUError::DeviceNotFound)
}

pub fn main() {
    // Список доступных устройств.
    let all_devices = Device::all();

    // Выводим список доступных устройств.
    println!("Available devices:");
    for device in &all_devices {
        println!(" - {} ({})", device.name(), device.vendor());
        println!("Memory: {} MB", device.memory() / 1024 / 1024);
        println!("Compute units: {}", device.compute_units());
        println!("Compute capability: {:?}", device.compute_capability());
    }

    // Первое доступное CUDA устройство.
    let cuda_device = get_cuda_device(all_devices).unwrap();
    println!("CUDA device: {}", cuda_device.name());

    
}