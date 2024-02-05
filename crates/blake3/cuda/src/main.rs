use rust_gpu_tools::{cuda, program_closures, Device, GPUError, Program, Vendor};

pub fn main() {
    // Список доступных устройств.
    let all_devices = Device::all();
    print_all_devices(&all_devices);

    // Первое доступное CUDA устройство.
    let cuda_device = get_cuda_device(all_devices).unwrap();
    println!("CUDA device: {}", cuda_device.name());

    // The test data to be hashed.
    let test_data: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];

    // This is the core. Here we write the interaction with the GPU independent of whether it is
    // CUDA or OpenCL.
    let closures = program_closures!(|program, _args| -> Result<Vec<u32>, GPUError> {
        // Copy the test data to the GPU.
        let data_buffer = program.create_buffer_from_slice(&test_data)?;

        // выводим содержимое буфера на экран
        let mut test_data_copy = test_data.clone();
        program.read_into_buffer(&data_buffer, &mut test_data_copy)?;
        println!("test_data_copy = {:?}\n", test_data_copy);

        // The result buffer has the same length as the input buffers.
        let result_buffer = unsafe { program.create_buffer::<u32>(test_data.len())? };
        print!("result_buffer = {:?}\n", result_buffer);

        // Get the kernel.
        //let test_data_len = test_data.len() as u32;
        let mut kernel = program.create_kernel("hello", 1, 1)?;
        kernel.run()?;

        kernel = program.create_kernel("sortDescending", 1, 1)?;
        kernel.arg(&data_buffer).arg(&result_buffer).run()?;

        let mut result = vec![0u32; test_data.len().try_into().unwrap()];
        program.read_into_buffer(&result_buffer, &mut result)?;

        Ok(result)
    }); //let closures =...

    // First we run it on CUDA if available
    let nv_dev_list = Device::by_vendor(Vendor::Nvidia);
    if !nv_dev_list.is_empty() {
        // Test NVIDIA CUDA Flow
        let cuda_program = cuda(nv_dev_list[0]);
        let cuda_result = cuda_program.run(closures, ()).unwrap();

        println!("CUDA result: {:?}", cuda_result);

        println!("CUDA test passed");
    } else {
        println!("No CUDA device found");
        println!("CUDA test skipped");
        println!();
    }
}

fn print_all_devices(all_devices: &Vec<&Device>) {
    // Выводим список доступных устройств.
    println!("Available devices:");
    for device in all_devices {
        println!(" - {} ({})", device.name(), device.vendor());
        println!("Memory: {} MB", device.memory() / 1024 / 1024);
        println!("Compute units: {}", device.compute_units());
        println!("Compute capability: {:?}", device.compute_capability());
    }
}

// Получаем первое доступное CUDA устройство из принимаемого списка.
fn get_cuda_device(devices: Vec<&Device>) -> Result<&Device, GPUError> {
    for device in devices {
        if device.vendor() == Vendor::Nvidia {
            return Ok(device);
        }
    }
    Err(GPUError::DeviceNotFound)
}

/// Returns a `Program` that runs on CUDA.
fn cuda(device: &Device) -> Program {
    // The kernel was compiled with:
    // nvcc -fatbin --x cu add.cl
    let cuda_kernel = include_bytes!("../blake3_cuda.fatbin");
    let cuda_device = device.cuda_device().unwrap();
    let cuda_program = cuda::Program::from_bytes(cuda_device, cuda_kernel).unwrap();
    Program::Cuda(cuda_program)
}

