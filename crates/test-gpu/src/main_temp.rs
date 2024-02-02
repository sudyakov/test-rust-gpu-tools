use rust_gpu_tools::{cuda, program_closures, Device, GPUError, Program, Vendor};
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
    let cuda_kernel = include_bytes!("../add/add.fatbin");
    let cuda_device = device.cuda_device().unwrap();
    let cuda_program = cuda::Program::from_bytes(cuda_device, cuda_kernel).unwrap();
    Program::Cuda(cuda_program)
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

    // Define some data that should be operated on.
    let aa: Vec<u32> = vec![1, 2, 3, 4];
    let bb: Vec<u32> = vec![5, 6, 7, 8];

    // This is the core. Here we write the interaction with the GPU independent of whether it is
    // CUDA or OpenCL.
    let closures = program_closures!(|program, _args| -> Result<Vec<u32>, GPUError> {
        // Make sure the input data has the same length.
        assert_eq!(aa.len(), bb.len());
        let length = aa.len();

        // Copy the data to the GPU.
        let aa_buffer = program.create_buffer_from_slice(&aa)?;
        let bb_buffer = program.create_buffer_from_slice(&bb)?;

        // The result buffer has the same length as the input buffers.
        let result_buffer = unsafe { program.create_buffer::<u32>(length)? };

        // Get the kernel.
        let kernel = program.create_kernel("add", 1, 1)?;

        // Execute the kernel.
        kernel
            .arg(&(length as u32))
            .arg(&aa_buffer)
            .arg(&bb_buffer)
            .arg(&result_buffer)
            .run()?;

        // Get the resulting data.
        let mut result = vec![0u32; length];
        program.read_into_buffer(&result_buffer, &mut result)?;

        Ok(result)
    });
    // First we run it on CUDA if available
    let nv_dev_list = Device::by_vendor(Vendor::Nvidia);
    if !nv_dev_list.is_empty() {
        // Test NVIDIA CUDA Flow
        let cuda_program = cuda(nv_dev_list[0]);
        let cuda_result = cuda_program.run(closures, ()).unwrap();
        assert_eq!(cuda_result, [6, 8, 10, 12]);
        println!("CUDA result: {:?}", cuda_result);

        println!("CUDA test passed");
    } else {
        println!("No CUDA device found");
        println!("CUDA test skipped");
        println!();
    }
}
