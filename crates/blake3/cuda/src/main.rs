use rust_gpu_tools::{cuda, program_closures, Device, GPUError, Program, Vendor};
use std::time::Instant;
use std::vec::*;

pub fn main() {
    let repeat_count = 10000;

    // NVIDIA GeForce RTX 4080 cores=9728, tmus=304, rops=112, memory=16Gb
    // total number of threads is global_work_size * local_work_size 
    let global_work_size = 1;
    let local_work_size = 1;

    let all_devices = Device::all();
    print_all_devices(&all_devices);

    let cuda_device = get_cuda_device(all_devices).unwrap();
    println!("CUDA device: {}", cuda_device.name());

    let test_data: Vec<u32> = vec![
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
        26, 27, 28, 29, 30, 31, 32,
    ];
    println!("test_data:\n{:?}\n", test_data);

    // Сортируем массив test_data по убыванию
    let closures = program_closures!(|program, _args| -> Result<Vec<u32>, GPUError> {
        let data_buffer = program.create_buffer_from_slice(&test_data)?;

        let result_buffer = unsafe { program.create_buffer::<u32>(test_data.len())? };

        let kernel = program.create_kernel("sortDescending", global_work_size, local_work_size)?;
        kernel
            .arg(&(test_data.len() as u32))
            .arg(&data_buffer)
            .arg(&result_buffer)
            .run()?;

        let mut result = vec![0u32; test_data.len()];
        program.read_into_buffer(&result_buffer, &mut result)?;

        Ok(result)
    });

    let cuda_program = cuda(cuda_device);

    // Начинаем отсчет времени
    let start = Instant::now();
    let mut cuda_result = cuda_program.run(closures, ());

    for _ in 0..repeat_count {
        cuda_result = cuda_program.run(closures, ());
    }

    // Замеряем время после выполнения функции
    let duration = start.elapsed();
    println!("CUDA result: \n{:?}", cuda_result.unwrap());
    // Выводим замеренное время
    println!("Time elapsed in expensive_function() is: {:?}", duration);
    println!("CUDA test passed\n");

    // Сортируем на процессоре
    let mut cpu_result: Vec<_> = test_data.clone().into_iter().collect();
    // Стартуем таймер для замера времени выполнения на CPU
    let cpu_start = Instant::now();

    for _ in 0..repeat_count {
        cpu_result.sort_by(|a, b| b.cmp(a));
    }

    //Выводим резульатт на укран
    println!("CPU result: \n{:?}", cpu_result);

    // Замеряем время после выполнения функции на процессоре
    let cpu_duration = cpu_start.elapsed();
    // Выводим замеренное время
    println!(
        "Time elapsed in expensive_function() is: {:?}",
        cpu_duration
    );
    println!("CPU test passed\n");

    fn print_all_devices(all_devices: &Vec<&Device>) {
        for device in all_devices {
            println!("Memory: {} MB", device.memory() / 1024 / 1024);
            println!("Compute capability: {:?}", device.compute_capability());
            
        }
    }

    fn get_cuda_device(devices: Vec<&Device>) -> Result<&Device, GPUError> {
        for device in devices {
            if device.vendor() == Vendor::Nvidia {
                return Ok(device);
            }
        }
        Err(GPUError::DeviceNotFound)
    }

    fn cuda(device: &Device) -> Program {
        let cuda_kernel = include_bytes!("../blake3_cuda.fatbin");
        let cuda_device = device.cuda_device().unwrap();
        let cuda_program = cuda::Program::from_bytes(cuda_device, cuda_kernel).unwrap();
        Program::Cuda(cuda_program)
    }
}
