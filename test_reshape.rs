use candle_core::{Tensor, Device, DType};

fn main() -> candle_core::Result<()> {
    let dev = Device::Cpu;
    let qkv = Tensor::arange(0f32, 24f32, &dev)?.reshape((1, 2, 12))?;
    println!("qkv strides: {:?}", qkv.stride());
    let q = qkv.narrow(2, 0, 8)?;
    println!("q strides: {:?}", q.stride());
    println!("q is_contiguous: {}", q.is_contiguous());
    let q_reshaped = q.reshape((1, 2, 4, 2));
    println!("q_reshaped result: {:?}", q_reshaped.is_ok());
    if let Err(e) = q_reshaped {
        println!("error: {}", e);
    }
    Ok(())
}
