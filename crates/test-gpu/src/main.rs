//
use rust_gpu_tools::{cuda, opencl, program_closures, Device, GPUError, Program, Vendor};

// Вывести на екран список доступных устройств с их описанием.
fn show_list_devices(devices: &Vec<&Device>) {
    println!("Available devices:");
    for device in devices {
        println!("- {} ({})", device.name(), device.vendor());
        println!("    Memory: {} MB", device.memory() / 1024 / 1024);
        println!("    Compute units: {}", device.compute_units());
        println!("    Compute capability: {:?}", device.compute_capability());
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let devices = Device::all();
    show_list_devices(&devices);

    // Найти первое CUDA устройство в списке devices
    let first_cude_device = devices.iter().find(|d| d.vendor() == Vendor::Nvidia);
    // Вывод имени первого устройства
    if let Some(device) = first_cude_device {
        println!("First CUDA device: {}", device.name());
    }
    Ok(())
} // end main function
