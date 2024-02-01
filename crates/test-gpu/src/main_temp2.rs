use rust_gpu_tools::{cuda, opencl, program_closures, Device, GPUError, Program, Vendor};

/// Returns a `Program` that runs on CUDA.
fn cuda(device: &Device) -> Program {
    // The kernel was compiled with:
    // nvcc -fatbin -gencode=arch=compute_52,code=sm_52 -gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_61,code=sm_61 -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_75,code=sm_75 -gencode=arch=compute_75,code=compute_75 --x cu add.cl
    let cuda_kernel = include_bytes!("bandwidthTest.fatbin");
    let cuda_device = device.cuda_device().unwrap();
    let cuda_program = cuda::Program::from_bytes(cuda_device, cuda_kernel).unwrap();
    Program::Cuda(cuda_program)
}

// /// Blak3 Returns a `Program` that run Blake3 Hash on CUDA.
// fn cuda_blake3_hash(device: &Device) -> Program {
//     // The kernel was compiled with:
//     // nvcc -fatbin -gencode=arch=compute_52,code=sm_52 -gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_61,code=sm_61 -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_75,code=sm_75 -gencode=arch=compute_75,code=compute_75 --x cu blake3_opcl.cl
//     let cuda_kernel = include_bytes!("blake3_gpu/blake3_cuda.fatbin");
//     let cuda_device = device.cuda_device().unwrap();
//     let cuda_program = cuda::Program::from_bytes(cuda_device, cuda_kernel).unwrap();
//     Program::Cuda(cuda_program)
// }

/// Returns a `Program` that runs on OpenCL.
fn opencl(device: &Device) -> Program {
    let opencl_kernel = include_str!("../../rust-gpu-tools/examples/add.cl");
    let opencl_device = device.opencl_device().unwrap();
    let opencl_program = opencl::Program::from_opencl(opencl_device, opencl_kernel).unwrap();
    Program::Opencl(opencl_program)
}

// // Test NVIDIA CUDA Flow Blake3
// fn test_nvidia_cuda_flow_blake3(
//     nv_dev_list: &Vec<&Device>,
//     closures_test_blake3: (
//         impl Fn(&cuda::Program, ()) -> Result<Vec<u32>, GPUError>,
//         impl Fn(&opencl::Program, ()) -> Result<Vec<u32>, GPUError>,
//     ),
// ) {
//     let cuda_program = cuda_blake3_hash(nv_dev_list[0]);
//     let cuda_result = cuda_program.run(closures_test_blake3, ()).expect("Не удалось получить CUDA устройство");
//     println!("Test NVIDIA CUDA Flow result: {:?}", cuda_result);
// }

// Test NVIDIA CUDA Flow
fn test_nvidia_cuda_flow(
    nv_dev_list: &Vec<&Device>,
    closures_test: (
        impl Fn(&cuda::Program, ()) -> Result<Vec<u32>, GPUError>,
        impl Fn(&opencl::Program, ()) -> Result<Vec<u32>, GPUError>,
    ),
) {
    let cuda_program = cuda(nv_dev_list[0]);
    let cuda_result = cuda_program.run(closures_test, ()).unwrap();
    assert_eq!(cuda_result, [6, 8, 10, 12]);
    println!("Test NVIDIA CUDA Flow result: {:?}", cuda_result);
}
// Test NVIDIA OpenCL Flow
fn test_nvidia_opencl_flow(
    nv_dev_list: Vec<&Device>,
    closures_test: (
        impl Fn(&cuda::Program, ()) -> Result<Vec<u32>, GPUError>,
        impl Fn(&opencl::Program, ()) -> Result<Vec<u32>, GPUError>,
    ),
) {
    let opencl_program = opencl(nv_dev_list[0]);
    let opencl_result = opencl_program.run(closures_test, ()).unwrap();
    assert_eq!(opencl_result, [6, 8, 10, 12]);
    println!("Test NVIDIA OpenCL Flow: {:?}", opencl_result);
}

// Вывести на екран список доступных устройств с их описанием.
fn show_list_devices() {
    let devices = Device::all();

    println!("Available devices:");
    for device in devices {
        println!("- {} ({})", device.name(), device.vendor());
        println!("    Memory: {} MB", device.memory() / 1024 / 1024);
        println!("    Compute units: {}", device.compute_units());
        println!("    Compute capability: {:?}", device.compute_capability());
    }
}

//
pub fn main() {
    // Вывести на екран список доступных устройств с их описанием.
    show_list_devices();

    print!("\nRun test-gpu: main.rs\n");
    // Define some data that should be operated on.
    // let aa: Vec<u32> = vec![1, 2, 3, 4];
    // let bb: Vec<u32> = vec![5, 6, 7, 8];
    let test_blake3: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32];

    // This is the core. Here we write the interaction with the GPU independent of whether it is
    // CUDA or OpenCL.
    let closures_test = program_closures!(|program, _args| -> Result<Vec<u32>, GPUError> {
        // Make sure the input data has the same length.
        assert_eq!(test_blake3.len(), 32);
        let length = test_blake3.len();

        // Copy the data to the GPU.
        // let aa_buffer = program.create_buffer_from_slice(&aa)?;
        // let bb_buffer = program.create_buffer_from_slice(&bb)?;
        let test_blake3_buffer = program.create_buffer_from_slice(&test_blake3)?;
        println!("Copy the data to the GPU - Done.\n");

        // The result buffer has the same length as the input buffers.
        let result_buffer = unsafe { program.create_buffer::<u32>(length)? };

        // Get the kernel.
        // let kernel = program.create_kernel("add", 1, 1)?;
        let kernel = program.create_kernel("cuda_blake3_hash", 1, 1)?;

        // Execute the kernel.
        print!("Execute the kernel started.\n");
        kernel
            .arg(&(length as u32))
            .arg(&test_blake3_buffer)
            .arg(&result_buffer)
            .run()?;

        // Get the resulting data.
        let mut result = vec![0u32; length];
        program.read_into_buffer(&result_buffer, &mut result)?;

        Ok(result)
    });

    // This is the core. Here we write the interaction with the GPU independent of whether it is
    // CUDA or OpenCL. Using blake3_cuda.fatbin
    // let closures_test_blake3 = program_closures!(|program, _args| -> Result<Vec<u32>, GPUError> {
    //     // Make sure the input data has the same length.
    //     assert_eq!(test_blake3.len(), 16);
    //     println!("{}",test_blake3.len());
    //     let length = test_blake3.len();

    //     // Copy the data to the GPU.
    //     let test_blake3_buffer = program.create_buffer_from_slice(&test_blake3)?;

    //     // The result buffer has the same length as the input buffers.
    //     let result_buffer = unsafe { program.create_buffer::<u32>(length)? };

    //     // Get the kernel.
    //     let kernel = program.create_kernel("blake3_hash", 1, 1)?;

    //     // Execute the kernel.
    //     print!("Execute the kernel started.\n");
    //     kernel
    //     .arg(&(length as u32))
    //     .arg(&test_blake3_buffer)
    //     .arg(&result_buffer)
    //     .run()?;

    // // Get the resulting data. 
    // let mut result = vec![0u32; length];
    // program.read_into_buffer(&result_buffer, &mut result)?;
    // Ok(result)
    // });

    // First we run it on CUDA if available
    let nv_dev_list = Device::by_vendor(Vendor::Nvidia);
    if !nv_dev_list.is_empty() {
        //
        test_nvidia_cuda_flow(&nv_dev_list, closures_test);
        //
        //test_nvidia_cuda_flow_blake3(&nv_dev_list, closures_test_blake3);
        //
        //test_nvidia_opencl_flow(nv_dev_list, closures_test);
    }


    // // Then we run it on Intel OpenCL if available
    // let intel_dev_list = Device::by_vendor(Vendor::Intel);
    // if !intel_dev_list.is_empty() {
    //     let opencl_program = opencl(intel_dev_list[0]);
    //     let opencl_result = opencl_program.run(closures_test, ()).unwrap();
    //     assert_eq!(opencl_result, [6, 8, 10, 12]);
    //     println!("OpenCL Intel result: {:?}", opencl_result);
    // }

    // let amd_dev_list = Device::by_vendor(Vendor::Amd);
    // if !amd_dev_list.is_empty() {
    //     let opencl_program = opencl(amd_dev_list[0]);
    //     let opencl_result = opencl_program.run(closures_test, ()).unwrap();
    //     assert_eq!(opencl_result, [6, 8, 10, 12]);
    //     println!("OpenCL Amd result: {:?}", opencl_result);
    // }
}