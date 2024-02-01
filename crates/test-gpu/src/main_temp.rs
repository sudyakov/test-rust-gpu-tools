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

pub fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Получить список доступных устройств
    let devices = Device::all();
    show_list_devices(&devices);

    // Найти первое CUDA устройство в списке devices
    let first_cude_device = devices
        .iter()
        .find(|d| d.vendor() == Vendor::Nvidia)
        .unwrap();
    // Вывод имени первого устройства
    println!("First CUDA device: {}", first_cude_device.name());

    // Define test data
    let test_data: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];

    // Вывест на экран занчение тестовых данных
println!("Test data: {:?}", test_data);



    //println!("Result: {:?}", result);

    Ok(())
}





