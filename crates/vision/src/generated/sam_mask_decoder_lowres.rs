// Generated from ONNX "../ml-bench/models/sam_mask_decoder_lowres.onnx" by burn-onnx
extern crate alloc;
use burn::prelude::*;
use burn::nn::LayerNorm;
use burn::nn::LayerNormConfig;
use burn::nn::Linear;
use burn::nn::LinearConfig;
use burn::nn::PaddingConfig2d;
use burn::nn::conv::Conv2d;
use burn::nn::conv::Conv2dConfig;
use burn::nn::conv::ConvTranspose2d;
use burn::nn::conv::ConvTranspose2dConfig;


#[derive(Module, Debug)]
pub struct Submodule1 {
    constant138: burn::module::Param<Tensor<1>>,
    constant139: burn::module::Param<Tensor<1>>,
    constant140: burn::module::Param<Tensor<1>>,
    constant141: burn::module::Param<Tensor<1>>,
    linear1: Linear,
    constant142: burn::module::Param<Tensor<1>>,
    constant90: burn::module::Param<Tensor<2>>,
    constant86: burn::module::Param<Tensor<2>>,
    constant87: burn::module::Param<Tensor<2>>,
    constant88: burn::module::Param<Tensor<2>>,
    constant89: burn::module::Param<Tensor<2>>,
    conv2d1: Conv2d,
    constant153: burn::module::Param<Tensor<1>>,
    constant154: burn::module::Param<Tensor<1>>,
    constant97: burn::module::Param<Tensor<3>>,
    constant98: burn::module::Param<Tensor<3>>,
    constant155: burn::module::Param<Tensor<1>>,
    constant156: burn::module::Param<Tensor<1>>,
    constant157: burn::module::Param<Tensor<1>>,
    conv2d2: Conv2d,
    constant158: burn::module::Param<Tensor<1>>,
    constant159: burn::module::Param<Tensor<1>>,
    constant99: burn::module::Param<Tensor<3>>,
    constant100: burn::module::Param<Tensor<3>>,
    constant160: burn::module::Param<Tensor<1>>,
    constant161: burn::module::Param<Tensor<1>>,
    constant162: burn::module::Param<Tensor<1>>,
    conv2d3: Conv2d,
    constant163: burn::module::Param<Tensor<1>>,
    constant101: burn::module::Param<Tensor<4>>,
    #[module(skip)]
    device: Device,
}
impl Submodule1 {
    #[allow(unused_variables)]
    pub fn new(device: &Device) -> Self {
        let constant138: burn::module::Param<Tensor<1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<
                1,
            >::from_data(
                burn::tensor::TensorData::from([0.5f64]),
                (device, burn::tensor::DType::F32),
            ),
            device.clone(),
            false,
            [1].into(),
        );
        let constant139: burn::module::Param<Tensor<1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<
                1,
            >::from_data(
                burn::tensor::TensorData::from([1024f64]),
                (device, burn::tensor::DType::F32),
            ),
            device.clone(),
            false,
            [1].into(),
        );
        let constant140: burn::module::Param<Tensor<1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<
                1,
            >::from_data(
                burn::tensor::TensorData::from([2f64]),
                (device, burn::tensor::DType::F32),
            ),
            device.clone(),
            false,
            [1].into(),
        );
        let constant141: burn::module::Param<Tensor<1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<
                1,
            >::from_data(
                burn::tensor::TensorData::from([1f64]),
                (device, burn::tensor::DType::F32),
            ),
            device.clone(),
            false,
            [1].into(),
        );
        let linear1 = LinearConfig::new(2, 128).with_bias(false).init(device);
        let constant142: burn::module::Param<Tensor<1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<
                1,
            >::from_data(
                burn::tensor::TensorData::from([6.2831854820251465f64]),
                (device, burn::tensor::DType::F32),
            ),
            device.clone(),
            false,
            [1].into(),
        );
        let constant90: burn::module::Param<Tensor<2>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<
                2,
            >::zeros([1, 256], (device, burn::tensor::DType::F32)),
            device.clone(),
            false,
            [1, 256].into(),
        );
        let constant86: burn::module::Param<Tensor<2>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<
                2,
            >::zeros([1, 256], (device, burn::tensor::DType::F32)),
            device.clone(),
            false,
            [1, 256].into(),
        );
        let constant87: burn::module::Param<Tensor<2>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<
                2,
            >::zeros([1, 256], (device, burn::tensor::DType::F32)),
            device.clone(),
            false,
            [1, 256].into(),
        );
        let constant88: burn::module::Param<Tensor<2>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<
                2,
            >::zeros([1, 256], (device, burn::tensor::DType::F32)),
            device.clone(),
            false,
            [1, 256].into(),
        );
        let constant89: burn::module::Param<Tensor<2>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<
                2,
            >::zeros([1, 256], (device, burn::tensor::DType::F32)),
            device.clone(),
            false,
            [1, 256].into(),
        );
        let conv2d1 = Conv2dConfig::new([1, 4], [2, 2])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let constant153: burn::module::Param<Tensor<1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<
                1,
            >::from_data(
                burn::tensor::TensorData::from([2f64]),
                (device, burn::tensor::DType::F32),
            ),
            device.clone(),
            false,
            [1].into(),
        );
        let constant154: burn::module::Param<Tensor<1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<
                1,
            >::from_data(
                burn::tensor::TensorData::from([0.0000009999999974752427f64]),
                (device, burn::tensor::DType::F32),
            ),
            device.clone(),
            false,
            [1].into(),
        );
        let constant97: burn::module::Param<Tensor<3>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<
                3,
            >::zeros([4, 1, 1], (device, burn::tensor::DType::F32)),
            device.clone(),
            false,
            [4, 1, 1].into(),
        );
        let constant98: burn::module::Param<Tensor<3>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<
                3,
            >::zeros([4, 1, 1], (device, burn::tensor::DType::F32)),
            device.clone(),
            false,
            [4, 1, 1].into(),
        );
        let constant155: burn::module::Param<Tensor<1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<
                1,
            >::from_data(
                burn::tensor::TensorData::from([1.4142135381698608f64]),
                (device, burn::tensor::DType::F32),
            ),
            device.clone(),
            false,
            [1].into(),
        );
        let constant156: burn::module::Param<Tensor<1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<
                1,
            >::from_data(
                burn::tensor::TensorData::from([1f64]),
                (device, burn::tensor::DType::F32),
            ),
            device.clone(),
            false,
            [1].into(),
        );
        let constant157: burn::module::Param<Tensor<1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<
                1,
            >::from_data(
                burn::tensor::TensorData::from([0.5f64]),
                (device, burn::tensor::DType::F32),
            ),
            device.clone(),
            false,
            [1].into(),
        );
        let conv2d2 = Conv2dConfig::new([4, 16], [2, 2])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let constant158: burn::module::Param<Tensor<1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<
                1,
            >::from_data(
                burn::tensor::TensorData::from([2f64]),
                (device, burn::tensor::DType::F32),
            ),
            device.clone(),
            false,
            [1].into(),
        );
        let constant159: burn::module::Param<Tensor<1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<
                1,
            >::from_data(
                burn::tensor::TensorData::from([0.0000009999999974752427f64]),
                (device, burn::tensor::DType::F32),
            ),
            device.clone(),
            false,
            [1].into(),
        );
        let constant99: burn::module::Param<Tensor<3>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<
                3,
            >::zeros([16, 1, 1], (device, burn::tensor::DType::F32)),
            device.clone(),
            false,
            [16, 1, 1].into(),
        );
        let constant100: burn::module::Param<Tensor<3>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<
                3,
            >::zeros([16, 1, 1], (device, burn::tensor::DType::F32)),
            device.clone(),
            false,
            [16, 1, 1].into(),
        );
        let constant160: burn::module::Param<Tensor<1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<
                1,
            >::from_data(
                burn::tensor::TensorData::from([1.4142135381698608f64]),
                (device, burn::tensor::DType::F32),
            ),
            device.clone(),
            false,
            [1].into(),
        );
        let constant161: burn::module::Param<Tensor<1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<
                1,
            >::from_data(
                burn::tensor::TensorData::from([1f64]),
                (device, burn::tensor::DType::F32),
            ),
            device.clone(),
            false,
            [1].into(),
        );
        let constant162: burn::module::Param<Tensor<1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<
                1,
            >::from_data(
                burn::tensor::TensorData::from([0.5f64]),
                (device, burn::tensor::DType::F32),
            ),
            device.clone(),
            false,
            [1].into(),
        );
        let conv2d3 = Conv2dConfig::new([16, 256], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let constant163: burn::module::Param<Tensor<1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<
                1,
            >::from_data(
                burn::tensor::TensorData::from([1f64]),
                (device, burn::tensor::DType::F32),
            ),
            device.clone(),
            false,
            [1].into(),
        );
        let constant101: burn::module::Param<Tensor<4>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<
                4,
            >::zeros([1, 256, 1, 1], (device, burn::tensor::DType::F32)),
            device.clone(),
            false,
            [1, 256, 1, 1].into(),
        );
        Self {
            constant138,
            constant139,
            constant140,
            constant141,
            linear1,
            constant142,
            constant90,
            constant86,
            constant87,
            constant88,
            constant89,
            conv2d1,
            constant153,
            constant154,
            constant97,
            constant98,
            constant155,
            constant156,
            constant157,
            conv2d2,
            constant158,
            constant159,
            constant99,
            constant100,
            constant160,
            constant161,
            constant162,
            conv2d3,
            constant163,
            constant101,
            device: device.clone(),
        }
    }
    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(
        &self,
        point_coords: Tensor<3>,
        point_labels: Tensor<2>,
        mask_input: Tensor<4>,
        has_mask_input: Tensor<1>,
    ) -> (Tensor<3>, Tensor<4>) {
        let constant138_out1 = self.constant138.val();
        let add1_out1 = point_coords
            .add((constant138_out1).unsqueeze_dims(&[0isize, 1isize]));
        let constant139_out1 = self.constant139.val();
        let div1_out1 = add1_out1
            .div((constant139_out1).unsqueeze_dims(&[0isize, 1isize]));
        let constant140_out1 = self.constant140.val();
        let mul1_out1 = div1_out1
            .mul((constant140_out1).unsqueeze_dims(&[0isize, 1isize]));
        let constant141_out1 = self.constant141.val();
        let sub1_out1 = mul1_out1
            .sub((constant141_out1).unsqueeze_dims(&[0isize, 1isize]));
        let linear1_out1 = self.linear1.forward(sub1_out1);
        let constant142_out1 = self.constant142.val();
        let mul2_out1 = linear1_out1
            .mul((constant142_out1).unsqueeze_dims(&[0isize, 1isize]));
        let sin1_out1 = mul2_out1.clone().sin();
        let cos1_out1 = mul2_out1.cos();
        let concat1_out1 = burn::tensor::Tensor::cat([sin1_out1, cos1_out1].into(), 2);
        let unsqueeze1_out1: Tensor<3> = point_labels.unsqueeze_dims::<3>(&[-1]);
        let shape1_out1: [i64; 3] = {
            let axes = &concat1_out1.clone().dims()[0..3];
            let mut output = [0i64; 3];
            for i in 0..3 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let expand1_out1 = {
            let onnx_shape: [i64; 3usize] = shape1_out1;
            let input_dims = unsqueeze1_out1.dims();
            let mut shape = onnx_shape;
            #[allow(clippy::needless_range_loop)]
            for i in 0..3usize {
                let dim_offset = i;
                if shape[dim_offset] == 1 && input_dims[i] > 1 {
                    shape[dim_offset] = input_dims[i] as i64;
                }
            }
            unsqueeze1_out1.expand(shape)
        };
        let constant144_out1 = -1f32;
        let equal1_out1 = expand1_out1.clone().equal_elem(constant144_out1);
        let not1_out1 = equal1_out1.bool_not();
        let cast1_out1 = not1_out1.float().cast(burn::tensor::DType::F32);
        let mul3_out1 = concat1_out1.mul(cast1_out1);
        let constant145_out1 = -1f32;
        let equal2_out1 = expand1_out1.clone().equal_elem(constant145_out1);
        let cast2_out1 = equal2_out1.float().cast(burn::tensor::DType::F32);
        let constant90_out1 = self.constant90.val();
        let mul4_out1 = (constant90_out1).unsqueeze_dims(&[0isize]).mul(cast2_out1);
        let add2_out1 = mul3_out1.add(mul4_out1);
        let constant147_out1 = 0f32;
        let equal3_out1 = expand1_out1.clone().equal_elem(constant147_out1);
        let cast3_out1 = equal3_out1.float().cast(burn::tensor::DType::F32);
        let constant86_out1 = self.constant86.val();
        let mul5_out1 = (constant86_out1).unsqueeze_dims(&[0isize]).mul(cast3_out1);
        let add3_out1 = add2_out1.add(mul5_out1);
        let constant148_out1 = 1f32;
        let equal4_out1 = expand1_out1.clone().equal_elem(constant148_out1);
        let cast4_out1 = equal4_out1.float().cast(burn::tensor::DType::F32);
        let constant87_out1 = self.constant87.val();
        let mul6_out1 = (constant87_out1).unsqueeze_dims(&[0isize]).mul(cast4_out1);
        let add4_out1 = add3_out1.add(mul6_out1);
        let constant150_out1 = 2f32;
        let equal5_out1 = expand1_out1.clone().equal_elem(constant150_out1);
        let cast5_out1 = equal5_out1.float().cast(burn::tensor::DType::F32);
        let constant88_out1 = self.constant88.val();
        let mul7_out1 = (constant88_out1).unsqueeze_dims(&[0isize]).mul(cast5_out1);
        let add5_out1 = add4_out1.add(mul7_out1);
        let constant152_out1 = 3f32;
        let equal6_out1 = expand1_out1.equal_elem(constant152_out1);
        let cast6_out1 = equal6_out1.float().cast(burn::tensor::DType::F32);
        let constant89_out1 = self.constant89.val();
        let mul8_out1 = (constant89_out1).unsqueeze_dims(&[0isize]).mul(cast6_out1);
        let add6_out1 = add5_out1.add(mul8_out1);
        let conv2d1_out1 = self.conv2d1.forward(mask_input);
        let reducemean1_out1 = { conv2d1_out1.clone().mean_dim(1usize) };
        let sub2_out1 = conv2d1_out1.sub(reducemean1_out1);
        let constant153_out1 = self.constant153.val();
        let pow1_out1 = sub2_out1
            .clone()
            .powf((constant153_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let reducemean2_out1 = { pow1_out1.mean_dim(1usize) };
        let constant154_out1 = self.constant154.val();
        let add7_out1 = reducemean2_out1
            .add((constant154_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let sqrt1_out1 = add7_out1.sqrt();
        let div2_out1 = sub2_out1.div(sqrt1_out1);
        let constant97_out1 = self.constant97.val();
        let mul9_out1 = (constant97_out1).unsqueeze_dims(&[0isize]).mul(div2_out1);
        let constant98_out1 = self.constant98.val();
        let add8_out1 = mul9_out1.add((constant98_out1).unsqueeze_dims(&[0isize]));
        let constant155_out1 = self.constant155.val();
        let div3_out1 = add8_out1
            .clone()
            .div((constant155_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let erf1_out1 = div3_out1.erf();
        let constant156_out1 = self.constant156.val();
        let add9_out1 = erf1_out1
            .add((constant156_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let mul10_out1 = add8_out1.mul(add9_out1);
        let constant157_out1 = self.constant157.val();
        let mul11_out1 = mul10_out1
            .mul((constant157_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let conv2d2_out1 = self.conv2d2.forward(mul11_out1);
        let reducemean3_out1 = { conv2d2_out1.clone().mean_dim(1usize) };
        let sub3_out1 = conv2d2_out1.sub(reducemean3_out1);
        let constant158_out1 = self.constant158.val();
        let pow2_out1 = sub3_out1
            .clone()
            .powf((constant158_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let reducemean4_out1 = { pow2_out1.mean_dim(1usize) };
        let constant159_out1 = self.constant159.val();
        let add10_out1 = reducemean4_out1
            .add((constant159_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let sqrt2_out1 = add10_out1.sqrt();
        let div4_out1 = sub3_out1.div(sqrt2_out1);
        let constant99_out1 = self.constant99.val();
        let mul12_out1 = (constant99_out1).unsqueeze_dims(&[0isize]).mul(div4_out1);
        let constant100_out1 = self.constant100.val();
        let add11_out1 = mul12_out1.add((constant100_out1).unsqueeze_dims(&[0isize]));
        let constant160_out1 = self.constant160.val();
        let div5_out1 = add11_out1
            .clone()
            .div((constant160_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let erf2_out1 = div5_out1.erf();
        let constant161_out1 = self.constant161.val();
        let add12_out1 = erf2_out1
            .add((constant161_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let mul13_out1 = add11_out1.mul(add12_out1);
        let constant162_out1 = self.constant162.val();
        let mul14_out1 = mul13_out1
            .mul((constant162_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let conv2d3_out1 = self.conv2d3.forward(mul14_out1);
        let mul15_out1 = (has_mask_input.clone())
            .unsqueeze_dims(&[0isize, 1isize, 2isize])
            .mul(conv2d3_out1);
        let constant163_out1 = self.constant163.val();
        let sub4_out1 = constant163_out1.sub(has_mask_input);
        let constant101_out1 = self.constant101.val();
        let mul16_out1 = (sub4_out1)
            .unsqueeze_dims(&[0isize, 1isize, 2isize])
            .mul(constant101_out1);
        let add13_out1 = mul15_out1.add(mul16_out1);
        (add6_out1, add13_out1)
    }
}
#[derive(Module, Debug)]
pub struct Submodule2 {
    constant164: burn::module::Param<Tensor<3>>,
    linear2: Linear,
    constant165: burn::module::Param<Tensor<1>>,
    constant172: burn::module::Param<Tensor<1, Int>>,
    constant102: burn::module::Param<Tensor<3>>,
    constant177: burn::module::Param<Tensor<1, Int>>,
    constant174: burn::module::Param<Tensor<1, Int>>,
    constant182: burn::module::Param<Tensor<1, Int>>,
    constant181: burn::module::Param<Tensor<1, Int>>,
    constant183: burn::module::Param<Tensor<1, Int>>,
    constant188: burn::module::Param<Tensor<1, Int>>,
    constant185: burn::module::Param<Tensor<1, Int>>,
    constant193: burn::module::Param<Tensor<1, Int>>,
    constant192: burn::module::Param<Tensor<1, Int>>,
    constant194: burn::module::Param<Tensor<1, Int>>,
    #[module(skip)]
    device: Device,
}
impl Submodule2 {
    #[allow(unused_variables)]
    pub fn new(device: &Device) -> Self {
        let constant164: burn::module::Param<Tensor<3>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<
                3,
            >::zeros([64, 64, 2], (device, burn::tensor::DType::F32)),
            device.clone(),
            false,
            [64, 64, 2].into(),
        );
        let linear2 = LinearConfig::new(2, 128).with_bias(false).init(device);
        let constant165: burn::module::Param<Tensor<1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<
                1,
            >::from_data(
                burn::tensor::TensorData::from([6.2831854820251465f64]),
                (device, burn::tensor::DType::F32),
            ),
            device.clone(),
            false,
            [1].into(),
        );
        let constant172: burn::module::Param<Tensor<1, Int>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<
                1,
                Int,
            >::from_data(
                burn::tensor::TensorData::from([-1i64]),
                (device, burn::tensor::DType::I64),
            ),
            device.clone(),
            false,
            [1].into(),
        );
        let constant102: burn::module::Param<Tensor<3>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<
                3,
            >::zeros([1, 5, 256], (device, burn::tensor::DType::F32)),
            device.clone(),
            false,
            [1, 5, 256].into(),
        );
        let constant177: burn::module::Param<Tensor<1, Int>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<
                1,
                Int,
            >::zeros([1], (device, burn::tensor::DType::I64)),
            device.clone(),
            false,
            [1].into(),
        );
        let constant174: burn::module::Param<Tensor<1, Int>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<
                1,
                Int,
            >::zeros([1], (device, burn::tensor::DType::I64)),
            device.clone(),
            false,
            [1].into(),
        );
        let constant182: burn::module::Param<Tensor<1, Int>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<
                1,
                Int,
            >::zeros([3], (device, burn::tensor::DType::I64)),
            device.clone(),
            false,
            [3].into(),
        );
        let constant181: burn::module::Param<Tensor<1, Int>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<
                1,
                Int,
            >::zeros([1], (device, burn::tensor::DType::I64)),
            device.clone(),
            false,
            [1].into(),
        );
        let constant183: burn::module::Param<Tensor<1, Int>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<
                1,
                Int,
            >::from_data(
                burn::tensor::TensorData::from([-1i64]),
                (device, burn::tensor::DType::I64),
            ),
            device.clone(),
            false,
            [1].into(),
        );
        let constant188: burn::module::Param<Tensor<1, Int>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<
                1,
                Int,
            >::zeros([1], (device, burn::tensor::DType::I64)),
            device.clone(),
            false,
            [1].into(),
        );
        let constant185: burn::module::Param<Tensor<1, Int>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<
                1,
                Int,
            >::zeros([1], (device, burn::tensor::DType::I64)),
            device.clone(),
            false,
            [1].into(),
        );
        let constant193: burn::module::Param<Tensor<1, Int>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<
                1,
                Int,
            >::zeros([3], (device, burn::tensor::DType::I64)),
            device.clone(),
            false,
            [3].into(),
        );
        let constant192: burn::module::Param<Tensor<1, Int>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<
                1,
                Int,
            >::zeros([1], (device, burn::tensor::DType::I64)),
            device.clone(),
            false,
            [1].into(),
        );
        let constant194: burn::module::Param<Tensor<1, Int>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<
                1,
                Int,
            >::from_data(
                burn::tensor::TensorData::from([-1i64]),
                (device, burn::tensor::DType::I64),
            ),
            device.clone(),
            false,
            [1].into(),
        );
        Self {
            constant164,
            linear2,
            constant165,
            constant172,
            constant102,
            constant177,
            constant174,
            constant182,
            constant181,
            constant183,
            constant188,
            constant185,
            constant193,
            constant192,
            constant194,
            device: device.clone(),
        }
    }
    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(
        &self,
        add6_out1: Tensor<3>,
        image_embeddings: Tensor<4>,
        add13_out1: Tensor<4>,
    ) -> (Tensor<5>, Tensor<4>, Tensor<3>) {
        let constant164_out1 = self.constant164.val();
        let linear2_out1 = self.linear2.forward(constant164_out1);
        let constant165_out1 = self.constant165.val();
        let mul17_out1 = linear2_out1
            .mul((constant165_out1).unsqueeze_dims(&[0isize, 1isize]));
        let sin2_out1 = mul17_out1.clone().sin();
        let cos2_out1 = mul17_out1.cos();
        let concat2_out1 = burn::tensor::Tensor::cat([sin2_out1, cos2_out1].into(), 2);
        let transpose1_out1 = concat2_out1.permute([2, 0, 1]);
        let unsqueeze2_out1: Tensor<4> = transpose1_out1.unsqueeze_dims::<4>(&[0]);
        let shape2_out1: [i64; 3] = {
            let axes = &add6_out1.clone().dims()[0..3];
            let mut output = [0i64; 3];
            for i in 0..3 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let gather1_out1 = shape2_out1[0] as i64;
        let unsqueeze3_out1 = [gather1_out1 as i64];
        let constant170_out1: [i64; 1] = [-1i64];
        let constant169_out1: [i64; 1] = [-1i64];
        let concat3_out1: [i64; 3usize] = [
            &unsqueeze3_out1[..],
            &constant169_out1[..],
            &constant170_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape1_out1 = concat3_out1;
        let shape3_out1: [i64; 1] = [3i64];
        let constantofshape1_out1 = Tensor::<
            1,
            Int,
        >::from_data(
                burn::tensor::TensorData::from([1i64 as i64]),
                (&self.device, burn::tensor::DType::I64),
            )
            .reshape([1])
            .expand(shape3_out1);
        let constant172_out1 = self.constant172.val();
        let mul18_out1 = constantofshape1_out1.clone().mul(constant172_out1);
        let equal7_out1 = Tensor::<
            1,
            burn::tensor::Int,
        >::from_data(
                burn::tensor::TensorData::from(&reshape1_out1 as &[i64]),
                (&self.device, burn::tensor::DType::I64),
            )
            .equal(mul18_out1);
        let where1_out1 = Tensor::<
            1,
            burn::tensor::Int,
        >::from_data(
                burn::tensor::TensorData::from(&reshape1_out1 as &[i64]),
                (&self.device, burn::tensor::DType::I64),
            )
            .mask_where(equal7_out1, constantofshape1_out1);
        let constant102_out1 = self.constant102.val();
        let expand2_out1 = {
            let onnx_shape: [i64; 3usize] = TryInto::<
                [i64; 3usize],
            >::try_into(where1_out1.to_data().convert::<i64>().as_slice().unwrap())
                .unwrap();
            let input_dims = constant102_out1.dims();
            let mut shape = onnx_shape;
            #[allow(clippy::needless_range_loop)]
            for i in 0..3usize {
                let dim_offset = i;
                if shape[dim_offset] == 1 && input_dims[i] > 1 {
                    shape[dim_offset] = input_dims[i] as i64;
                }
            }
            constant102_out1.expand(shape)
        };
        let concat4_out1 = burn::tensor::Tensor::cat(
            [expand2_out1, add6_out1].into(),
            1,
        );
        let shape4_out1: [i64; 3] = {
            let axes = &concat4_out1.clone().dims()[0..3];
            let mut output = [0i64; 3];
            for i in 0..3 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let gather2_out1 = shape4_out1[0] as i64;
        let constantofshape2_out1: [i64; 1usize] = [1i64];
        let constant176_out1 = -1i64;
        let mul19_out1 = {
            let mut result = constantofshape2_out1;
            let scalar = constant176_out1;
            for result_item in result.iter_mut() {
                *result_item = result_item.saturating_mul(scalar);
            }
            result
        };
        let constant177_out1 = self.constant177.val();
        let equal8_out1 = constant177_out1
            .equal(
                Tensor::<
                    1,
                    burn::tensor::Int,
                >::from_data(
                    burn::tensor::TensorData::from(&mul19_out1 as &[i64]),
                    (&self.device, burn::tensor::DType::I64),
                ),
            );
        let constant174_out1 = self.constant174.val();
        let where2_out1 = constant174_out1
            .mask_where(
                equal8_out1,
                Tensor::<
                    1,
                    burn::tensor::Int,
                >::from_data(
                    burn::tensor::TensorData::from(&constantofshape2_out1 as &[i64]),
                    (&self.device, burn::tensor::DType::I64),
                ),
            );
        let expand3_out1 = {
            let input = Tensor::<
                1,
                Int,
            >::from_data(
                burn::tensor::TensorData::from([gather2_out1]),
                (&self.device, burn::tensor::DType::I64),
            );
            input
                .expand(
                    TryInto::<
                        [i64; 1usize],
                    >::try_into(
                            where2_out1.to_data().convert::<i64>().as_slice().unwrap(),
                        )
                        .unwrap(),
                )
        };
        let unsqueeze4_out1: Tensor<5> = image_embeddings.unsqueeze_dims::<5>(&[1]);
        let constant182_out1 = self.constant182.val();
        let constant181_out1 = self.constant181.val();
        let concat5_out1 = burn::tensor::Tensor::cat(
            [constant181_out1, expand3_out1, constant182_out1].into(),
            0,
        );
        let shape5_out1: [i64; 1] = {
            let axes = &concat5_out1.clone().dims()[0..1];
            let mut output = [0i64; 1];
            for i in 0..1 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let constantofshape3_out1 = Tensor::<
            1,
            Int,
        >::from_data(
                burn::tensor::TensorData::from([1i64 as i64]),
                (&self.device, burn::tensor::DType::I64),
            )
            .reshape([1])
            .expand(shape5_out1);
        let constant183_out1 = self.constant183.val();
        let mul20_out1 = constantofshape3_out1.clone().mul(constant183_out1);
        let equal9_out1 = concat5_out1.clone().equal(mul20_out1);
        let where3_out1 = concat5_out1.mask_where(equal9_out1, constantofshape3_out1);
        let expand4_out1 = {
            let onnx_shape: [i64; 5usize] = TryInto::<
                [i64; 5usize],
            >::try_into(where3_out1.to_data().convert::<i64>().as_slice().unwrap())
                .unwrap();
            let input_dims = unsqueeze4_out1.dims();
            let mut shape = onnx_shape;
            #[allow(clippy::needless_range_loop)]
            for i in 0..5usize {
                let dim_offset = i;
                if shape[dim_offset] == 1 && input_dims[i] > 1 {
                    shape[dim_offset] = input_dims[i] as i64;
                }
            }
            unsqueeze4_out1.expand(shape)
        };
        let reshape2_out1 = expand4_out1.reshape([-1, 256, 64, 64]);
        let add14_out1 = reshape2_out1.add(add13_out1);
        let constantofshape4_out1: [i64; 1usize] = [1i64];
        let constant187_out1 = -1i64;
        let mul21_out1 = {
            let mut result = constantofshape4_out1;
            let scalar = constant187_out1;
            for result_item in result.iter_mut() {
                *result_item = result_item.saturating_mul(scalar);
            }
            result
        };
        let constant188_out1 = self.constant188.val();
        let equal10_out1 = constant188_out1
            .equal(
                Tensor::<
                    1,
                    burn::tensor::Int,
                >::from_data(
                    burn::tensor::TensorData::from(&mul21_out1 as &[i64]),
                    (&self.device, burn::tensor::DType::I64),
                ),
            );
        let constant185_out1 = self.constant185.val();
        let where4_out1 = constant185_out1
            .mask_where(
                equal10_out1,
                Tensor::<
                    1,
                    burn::tensor::Int,
                >::from_data(
                    burn::tensor::TensorData::from(&constantofshape4_out1 as &[i64]),
                    (&self.device, burn::tensor::DType::I64),
                ),
            );
        let expand5_out1 = {
            let input = Tensor::<
                1,
                Int,
            >::from_data(
                burn::tensor::TensorData::from([gather2_out1]),
                (&self.device, burn::tensor::DType::I64),
            );
            input
                .expand(
                    TryInto::<
                        [i64; 1usize],
                    >::try_into(
                            where4_out1.to_data().convert::<i64>().as_slice().unwrap(),
                        )
                        .unwrap(),
                )
        };
        let unsqueeze5_out1: Tensor<5> = unsqueeze2_out1.unsqueeze_dims::<5>(&[1]);
        let constant193_out1 = self.constant193.val();
        let constant192_out1 = self.constant192.val();
        let concat7_out1 = burn::tensor::Tensor::cat(
            [constant192_out1, expand5_out1, constant193_out1].into(),
            0,
        );
        let shape6_out1: [i64; 1] = {
            let axes = &concat7_out1.clone().dims()[0..1];
            let mut output = [0i64; 1];
            for i in 0..1 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let constantofshape5_out1 = Tensor::<
            1,
            Int,
        >::from_data(
                burn::tensor::TensorData::from([1i64 as i64]),
                (&self.device, burn::tensor::DType::I64),
            )
            .reshape([1])
            .expand(shape6_out1);
        let constant194_out1 = self.constant194.val();
        let mul22_out1 = constantofshape5_out1.clone().mul(constant194_out1);
        let equal11_out1 = concat7_out1.clone().equal(mul22_out1);
        let where5_out1 = concat7_out1.mask_where(equal11_out1, constantofshape5_out1);
        let expand6_out1 = {
            let onnx_shape: [i64; 5usize] = TryInto::<
                [i64; 5usize],
            >::try_into(where5_out1.to_data().convert::<i64>().as_slice().unwrap())
                .unwrap();
            let input_dims = unsqueeze5_out1.dims();
            let mut shape = onnx_shape;
            #[allow(clippy::needless_range_loop)]
            for i in 0..5usize {
                let dim_offset = i;
                if shape[dim_offset] == 1 && input_dims[i] > 1 {
                    shape[dim_offset] = input_dims[i] as i64;
                }
            }
            unsqueeze5_out1.expand(shape)
        };
        (expand6_out1, add14_out1, concat4_out1)
    }
}
#[derive(Module, Debug)]
pub struct Submodule3 {
    linear3: Linear,
    linear4: Linear,
    linear5: Linear,
    constant232: burn::module::Param<Tensor<1>>,
    #[module(skip)]
    device: Device,
}
impl Submodule3 {
    #[allow(unused_variables)]
    pub fn new(device: &Device) -> Self {
        let linear3 = LinearConfig::new(256, 256).with_bias(true).init(device);
        let linear4 = LinearConfig::new(256, 256).with_bias(true).init(device);
        let linear5 = LinearConfig::new(256, 256).with_bias(true).init(device);
        let constant232: burn::module::Param<Tensor<1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<
                1,
            >::from_data(
                burn::tensor::TensorData::from([5.656854152679443f64]),
                (device, burn::tensor::DType::F32),
            ),
            device.clone(),
            false,
            [1].into(),
        );
        Self {
            linear3,
            linear4,
            linear5,
            constant232,
            device: device.clone(),
        }
    }
    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(
        &self,
        expand6_out1: Tensor<5>,
        add14_out1: Tensor<4>,
        concat4_out1: Tensor<3>,
    ) -> (Tensor<4>, Tensor<3>, Tensor<3>, i64, i64, i64, i64) {
        let reshape3_out1 = expand6_out1.reshape([-1, 256, 64, 64]);
        let shape7_out1: [i64; 4] = {
            let axes = &add14_out1.clone().dims()[0..4];
            let mut output = [0i64; 4];
            for i in 0..4 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let gather3_out1 = shape7_out1[0] as i64;
        let gather4_out1 = shape7_out1[1] as i64;
        let gather5_out1 = shape7_out1[2] as i64;
        let gather6_out1 = shape7_out1[3] as i64;
        let slice1_out1: [i64; 2] = shape7_out1[0..2].try_into().unwrap();
        let constant203_out1: [i64; 1] = [-1i64];
        let concat9_out1: [i64; 3usize] = [&slice1_out1[..], &constant203_out1[..]]
            .concat()
            .try_into()
            .unwrap();
        let reshape4_out1 = add14_out1.reshape(concat9_out1);
        let transpose2_out1 = reshape4_out1.permute([0, 2, 1]);
        let shape12_out1: [i64; 4] = {
            let axes = &reshape3_out1.clone().dims()[0..4];
            let mut output = [0i64; 4];
            for i in 0..4 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let slice2_out1: [i64; 2] = shape12_out1[0..2].try_into().unwrap();
        let constant207_out1: [i64; 1] = [-1i64];
        let concat10_out1: [i64; 3usize] = [&slice2_out1[..], &constant207_out1[..]]
            .concat()
            .try_into()
            .unwrap();
        let reshape5_out1 = reshape3_out1.reshape(concat10_out1);
        let transpose3_out1 = reshape5_out1.permute([0, 2, 1]);
        let linear3_out1 = self.linear3.forward(concat4_out1.clone());
        let linear4_out1 = self.linear4.forward(concat4_out1.clone());
        let linear5_out1 = self.linear5.forward(concat4_out1);
        let shape13_out1: [i64; 3] = {
            let axes = &linear3_out1.clone().dims()[0..3];
            let mut output = [0i64; 3];
            for i in 0..3 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let gather7_out1 = shape13_out1[0] as i64;
        let gather8_out1 = shape13_out1[1] as i64;
        let gather9_out1 = shape13_out1[2] as i64;
        let constant211_out1 = 8i64;
        let div6_out1 = gather9_out1 / constant211_out1;
        let unsqueeze6_out1 = [gather7_out1 as i64];
        let unsqueeze7_out1 = [gather8_out1 as i64];
        let unsqueeze8_out1 = [div6_out1 as i64];
        let constant214_out1: [i64; 1] = [8i64];
        let concat11_out1: [i64; 4usize] = [
            &unsqueeze6_out1[..],
            &unsqueeze7_out1[..],
            &constant214_out1[..],
            &unsqueeze8_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape6_out1 = linear3_out1.reshape(concat11_out1);
        let transpose4_out1 = reshape6_out1.permute([0, 2, 1, 3]);
        let shape16_out1: [i64; 3] = {
            let axes = &linear4_out1.clone().dims()[0..3];
            let mut output = [0i64; 3];
            for i in 0..3 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let gather10_out1 = shape16_out1[0] as i64;
        let gather11_out1 = shape16_out1[1] as i64;
        let gather12_out1 = shape16_out1[2] as i64;
        let constant219_out1 = 8i64;
        let div7_out1 = gather12_out1 / constant219_out1;
        let unsqueeze9_out1 = [gather10_out1 as i64];
        let unsqueeze10_out1 = [gather11_out1 as i64];
        let unsqueeze11_out1 = [div7_out1 as i64];
        let constant222_out1: [i64; 1] = [8i64];
        let concat12_out1: [i64; 4usize] = [
            &unsqueeze9_out1[..],
            &unsqueeze10_out1[..],
            &constant222_out1[..],
            &unsqueeze11_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape7_out1 = linear4_out1.reshape(concat12_out1);
        let shape19_out1: [i64; 3] = {
            let axes = &linear5_out1.clone().dims()[0..3];
            let mut output = [0i64; 3];
            for i in 0..3 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let gather13_out1 = shape19_out1[0] as i64;
        let gather14_out1 = shape19_out1[1] as i64;
        let gather15_out1 = shape19_out1[2] as i64;
        let constant227_out1 = 8i64;
        let div8_out1 = gather15_out1 / constant227_out1;
        let unsqueeze12_out1 = [gather13_out1 as i64];
        let unsqueeze13_out1 = [gather14_out1 as i64];
        let unsqueeze14_out1 = [div8_out1 as i64];
        let constant230_out1: [i64; 1] = [8i64];
        let concat13_out1: [i64; 4usize] = [
            &unsqueeze12_out1[..],
            &unsqueeze13_out1[..],
            &constant230_out1[..],
            &unsqueeze14_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape8_out1 = linear5_out1.reshape(concat13_out1);
        let transpose5_out1 = reshape8_out1.permute([0, 2, 1, 3]);
        let transpose6_out1 = reshape7_out1.permute([0, 2, 3, 1]);
        let matmul6_out1 = transpose4_out1.matmul(transpose6_out1);
        let constant232_out1 = self.constant232.val();
        let div9_out1 = matmul6_out1
            .div((constant232_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let softmax1_out1 = burn::tensor::activation::softmax(div9_out1, 3);
        let matmul7_out1 = softmax1_out1.matmul(transpose5_out1);
        (
            matmul7_out1,
            transpose2_out1,
            transpose3_out1,
            gather3_out1,
            gather4_out1,
            gather5_out1,
            gather6_out1,
        )
    }
}
#[derive(Module, Debug)]
pub struct Submodule4 {
    linear6: Linear,
    layernormalization1: LayerNorm,
    linear7: Linear,
    linear8: Linear,
    linear9: Linear,
    constant264: burn::module::Param<Tensor<1>>,
    linear10: Linear,
    layernormalization2: LayerNorm,
    linear11: Linear,
    linear12: Linear,
    layernormalization3: LayerNorm,
    linear13: Linear,
    linear14: Linear,
    linear15: Linear,
    constant296: burn::module::Param<Tensor<1>>,
    linear16: Linear,
    layernormalization4: LayerNorm,
    linear17: Linear,
    linear18: Linear,
    linear19: Linear,
    constant328: burn::module::Param<Tensor<1>>,
    linear20: Linear,
    #[module(skip)]
    device: Device,
}
impl Submodule4 {
    #[allow(unused_variables)]
    pub fn new(device: &Device) -> Self {
        let linear6 = LinearConfig::new(256, 256).with_bias(true).init(device);
        let layernormalization1 = LayerNormConfig::new(256)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(true)
            .init(device);
        let linear7 = LinearConfig::new(256, 128).with_bias(true).init(device);
        let linear8 = LinearConfig::new(256, 128).with_bias(true).init(device);
        let linear9 = LinearConfig::new(256, 128).with_bias(true).init(device);
        let constant264: burn::module::Param<Tensor<1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<
                1,
            >::from_data(
                burn::tensor::TensorData::from([4f64]),
                (device, burn::tensor::DType::F32),
            ),
            device.clone(),
            false,
            [1].into(),
        );
        let linear10 = LinearConfig::new(128, 256).with_bias(true).init(device);
        let layernormalization2 = LayerNormConfig::new(256)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(true)
            .init(device);
        let linear11 = LinearConfig::new(256, 2048).with_bias(true).init(device);
        let linear12 = LinearConfig::new(2048, 256).with_bias(true).init(device);
        let layernormalization3 = LayerNormConfig::new(256)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(true)
            .init(device);
        let linear13 = LinearConfig::new(256, 128).with_bias(true).init(device);
        let linear14 = LinearConfig::new(256, 128).with_bias(true).init(device);
        let linear15 = LinearConfig::new(256, 128).with_bias(true).init(device);
        let constant296: burn::module::Param<Tensor<1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<
                1,
            >::from_data(
                burn::tensor::TensorData::from([4f64]),
                (device, burn::tensor::DType::F32),
            ),
            device.clone(),
            false,
            [1].into(),
        );
        let linear16 = LinearConfig::new(128, 256).with_bias(true).init(device);
        let layernormalization4 = LayerNormConfig::new(256)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(true)
            .init(device);
        let linear17 = LinearConfig::new(256, 256).with_bias(true).init(device);
        let linear18 = LinearConfig::new(256, 256).with_bias(true).init(device);
        let linear19 = LinearConfig::new(256, 256).with_bias(true).init(device);
        let constant328: burn::module::Param<Tensor<1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<
                1,
            >::from_data(
                burn::tensor::TensorData::from([5.656854152679443f64]),
                (device, burn::tensor::DType::F32),
            ),
            device.clone(),
            false,
            [1].into(),
        );
        let linear20 = LinearConfig::new(256, 256).with_bias(true).init(device);
        Self {
            linear6,
            layernormalization1,
            linear7,
            linear8,
            linear9,
            constant264,
            linear10,
            layernormalization2,
            linear11,
            linear12,
            layernormalization3,
            linear13,
            linear14,
            linear15,
            constant296,
            linear16,
            layernormalization4,
            linear17,
            linear18,
            linear19,
            constant328,
            linear20,
            device: device.clone(),
        }
    }
    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(
        &self,
        matmul7_out1: Tensor<4>,
        concat4_out1: Tensor<3>,
        transpose2_out1: Tensor<3>,
        transpose3_out1: Tensor<3>,
    ) -> (Tensor<3>, Tensor<3>) {
        let shape22_out1: [i64; 4] = {
            let axes = &matmul7_out1.clone().dims()[0..4];
            let mut output = [0i64; 4];
            for i in 0..4 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let gather16_out1 = shape22_out1[0] as i64;
        let gather17_out1 = shape22_out1[1] as i64;
        let gather18_out1 = shape22_out1[2] as i64;
        let gather19_out1 = shape22_out1[3] as i64;
        let transpose7_out1 = matmul7_out1.permute([0, 2, 1, 3]);
        let mul23_out1 = gather17_out1 * gather19_out1;
        let unsqueeze15_out1 = [gather16_out1 as i64];
        let unsqueeze16_out1 = [gather18_out1 as i64];
        let unsqueeze17_out1 = [mul23_out1 as i64];
        let concat14_out1: [i64; 3usize] = [
            &unsqueeze15_out1[..],
            &unsqueeze16_out1[..],
            &unsqueeze17_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape9_out1 = transpose7_out1.reshape(concat14_out1);
        let linear6_out1 = self.linear6.forward(reshape9_out1);
        let layernormalization1_out1 = {
            let dtype = linear6_out1.dtype();
            self.layernormalization1
                .forward(linear6_out1.cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let add15_out1 = layernormalization1_out1.clone().add(concat4_out1.clone());
        let add16_out1 = transpose2_out1.clone().add(transpose3_out1);
        let linear7_out1 = self.linear7.forward(add15_out1);
        let linear8_out1 = self.linear8.forward(add16_out1.clone());
        let linear9_out1 = self.linear9.forward(transpose2_out1.clone());
        let shape26_out1: [i64; 3] = {
            let axes = &linear7_out1.clone().dims()[0..3];
            let mut output = [0i64; 3];
            for i in 0..3 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let gather20_out1 = shape26_out1[0] as i64;
        let gather21_out1 = shape26_out1[1] as i64;
        let gather22_out1 = shape26_out1[2] as i64;
        let constant243_out1 = 8i64;
        let div10_out1 = gather22_out1 / constant243_out1;
        let unsqueeze18_out1 = [gather20_out1 as i64];
        let unsqueeze19_out1 = [gather21_out1 as i64];
        let unsqueeze20_out1 = [div10_out1 as i64];
        let constant246_out1: [i64; 1] = [8i64];
        let concat15_out1: [i64; 4usize] = [
            &unsqueeze18_out1[..],
            &unsqueeze19_out1[..],
            &constant246_out1[..],
            &unsqueeze20_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape10_out1 = linear7_out1.reshape(concat15_out1);
        let transpose8_out1 = reshape10_out1.permute([0, 2, 1, 3]);
        let shape29_out1: [i64; 3] = {
            let axes = &linear8_out1.clone().dims()[0..3];
            let mut output = [0i64; 3];
            for i in 0..3 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let gather23_out1 = shape29_out1[0] as i64;
        let gather24_out1 = shape29_out1[1] as i64;
        let gather25_out1 = shape29_out1[2] as i64;
        let constant251_out1 = 8i64;
        let div11_out1 = gather25_out1 / constant251_out1;
        let unsqueeze21_out1 = [gather23_out1 as i64];
        let unsqueeze22_out1 = [gather24_out1 as i64];
        let unsqueeze23_out1 = [div11_out1 as i64];
        let constant254_out1: [i64; 1] = [8i64];
        let concat16_out1: [i64; 4usize] = [
            &unsqueeze21_out1[..],
            &unsqueeze22_out1[..],
            &constant254_out1[..],
            &unsqueeze23_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape11_out1 = linear8_out1.reshape(concat16_out1);
        let shape32_out1: [i64; 3] = {
            let axes = &linear9_out1.clone().dims()[0..3];
            let mut output = [0i64; 3];
            for i in 0..3 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let gather26_out1 = shape32_out1[0] as i64;
        let gather27_out1 = shape32_out1[1] as i64;
        let gather28_out1 = shape32_out1[2] as i64;
        let constant259_out1 = 8i64;
        let div12_out1 = gather28_out1 / constant259_out1;
        let unsqueeze24_out1 = [gather26_out1 as i64];
        let unsqueeze25_out1 = [gather27_out1 as i64];
        let unsqueeze26_out1 = [div12_out1 as i64];
        let constant262_out1: [i64; 1] = [8i64];
        let concat17_out1: [i64; 4usize] = [
            &unsqueeze24_out1[..],
            &unsqueeze25_out1[..],
            &constant262_out1[..],
            &unsqueeze26_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape12_out1 = linear9_out1.reshape(concat17_out1);
        let transpose9_out1 = reshape12_out1.permute([0, 2, 1, 3]);
        let transpose10_out1 = reshape11_out1.permute([0, 2, 3, 1]);
        let matmul12_out1 = transpose8_out1.matmul(transpose10_out1);
        let constant264_out1 = self.constant264.val();
        let div13_out1 = matmul12_out1
            .div((constant264_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let softmax2_out1 = burn::tensor::activation::softmax(div13_out1, 3);
        let matmul13_out1 = softmax2_out1.matmul(transpose9_out1);
        let shape35_out1: [i64; 4] = {
            let axes = &matmul13_out1.clone().dims()[0..4];
            let mut output = [0i64; 4];
            for i in 0..4 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let gather29_out1 = shape35_out1[0] as i64;
        let gather30_out1 = shape35_out1[1] as i64;
        let gather31_out1 = shape35_out1[2] as i64;
        let gather32_out1 = shape35_out1[3] as i64;
        let transpose11_out1 = matmul13_out1.permute([0, 2, 1, 3]);
        let mul24_out1 = gather30_out1 * gather32_out1;
        let unsqueeze27_out1 = [gather29_out1 as i64];
        let unsqueeze28_out1 = [gather31_out1 as i64];
        let unsqueeze29_out1 = [mul24_out1 as i64];
        let concat18_out1: [i64; 3usize] = [
            &unsqueeze27_out1[..],
            &unsqueeze28_out1[..],
            &unsqueeze29_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape13_out1 = transpose11_out1.reshape(concat18_out1);
        let linear10_out1 = self.linear10.forward(reshape13_out1);
        let add17_out1 = layernormalization1_out1.add(linear10_out1);
        let layernormalization2_out1 = {
            let dtype = add17_out1.dtype();
            self.layernormalization2
                .forward(add17_out1.cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let linear11_out1 = self.linear11.forward(layernormalization2_out1.clone());
        let relu1_out1 = burn::tensor::activation::relu(linear11_out1);
        let linear12_out1 = self.linear12.forward(relu1_out1);
        let add18_out1 = layernormalization2_out1.add(linear12_out1);
        let layernormalization3_out1 = {
            let dtype = add18_out1.dtype();
            self.layernormalization3
                .forward(add18_out1.cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let add19_out1 = layernormalization3_out1.clone().add(concat4_out1);
        let linear13_out1 = self.linear13.forward(add16_out1);
        let linear14_out1 = self.linear14.forward(add19_out1.clone());
        let linear15_out1 = self.linear15.forward(layernormalization3_out1.clone());
        let shape39_out1: [i64; 3] = {
            let axes = &linear13_out1.clone().dims()[0..3];
            let mut output = [0i64; 3];
            for i in 0..3 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let gather33_out1 = shape39_out1[0] as i64;
        let gather34_out1 = shape39_out1[1] as i64;
        let gather35_out1 = shape39_out1[2] as i64;
        let constant275_out1 = 8i64;
        let div14_out1 = gather35_out1 / constant275_out1;
        let unsqueeze30_out1 = [gather33_out1 as i64];
        let unsqueeze31_out1 = [gather34_out1 as i64];
        let unsqueeze32_out1 = [div14_out1 as i64];
        let constant278_out1: [i64; 1] = [8i64];
        let concat19_out1: [i64; 4usize] = [
            &unsqueeze30_out1[..],
            &unsqueeze31_out1[..],
            &constant278_out1[..],
            &unsqueeze32_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape14_out1 = linear13_out1.reshape(concat19_out1);
        let transpose12_out1 = reshape14_out1.permute([0, 2, 1, 3]);
        let shape42_out1: [i64; 3] = {
            let axes = &linear14_out1.clone().dims()[0..3];
            let mut output = [0i64; 3];
            for i in 0..3 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let gather36_out1 = shape42_out1[0] as i64;
        let gather37_out1 = shape42_out1[1] as i64;
        let gather38_out1 = shape42_out1[2] as i64;
        let constant283_out1 = 8i64;
        let div15_out1 = gather38_out1 / constant283_out1;
        let unsqueeze33_out1 = [gather36_out1 as i64];
        let unsqueeze34_out1 = [gather37_out1 as i64];
        let unsqueeze35_out1 = [div15_out1 as i64];
        let constant286_out1: [i64; 1] = [8i64];
        let concat20_out1: [i64; 4usize] = [
            &unsqueeze33_out1[..],
            &unsqueeze34_out1[..],
            &constant286_out1[..],
            &unsqueeze35_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape15_out1 = linear14_out1.reshape(concat20_out1);
        let shape45_out1: [i64; 3] = {
            let axes = &linear15_out1.clone().dims()[0..3];
            let mut output = [0i64; 3];
            for i in 0..3 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let gather39_out1 = shape45_out1[0] as i64;
        let gather40_out1 = shape45_out1[1] as i64;
        let gather41_out1 = shape45_out1[2] as i64;
        let constant291_out1 = 8i64;
        let div16_out1 = gather41_out1 / constant291_out1;
        let unsqueeze36_out1 = [gather39_out1 as i64];
        let unsqueeze37_out1 = [gather40_out1 as i64];
        let unsqueeze38_out1 = [div16_out1 as i64];
        let constant294_out1: [i64; 1] = [8i64];
        let concat21_out1: [i64; 4usize] = [
            &unsqueeze36_out1[..],
            &unsqueeze37_out1[..],
            &constant294_out1[..],
            &unsqueeze38_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape16_out1 = linear15_out1.reshape(concat21_out1);
        let transpose13_out1 = reshape16_out1.permute([0, 2, 1, 3]);
        let transpose14_out1 = reshape15_out1.permute([0, 2, 3, 1]);
        let matmul20_out1 = transpose12_out1.matmul(transpose14_out1);
        let constant296_out1 = self.constant296.val();
        let div17_out1 = matmul20_out1
            .div((constant296_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let softmax3_out1 = burn::tensor::activation::softmax(div17_out1, 3);
        let matmul21_out1 = softmax3_out1.matmul(transpose13_out1);
        let shape48_out1: [i64; 4] = {
            let axes = &matmul21_out1.clone().dims()[0..4];
            let mut output = [0i64; 4];
            for i in 0..4 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let gather42_out1 = shape48_out1[0] as i64;
        let gather43_out1 = shape48_out1[1] as i64;
        let gather44_out1 = shape48_out1[2] as i64;
        let gather45_out1 = shape48_out1[3] as i64;
        let transpose15_out1 = matmul21_out1.permute([0, 2, 1, 3]);
        let mul25_out1 = gather43_out1 * gather45_out1;
        let unsqueeze39_out1 = [gather42_out1 as i64];
        let unsqueeze40_out1 = [gather44_out1 as i64];
        let unsqueeze41_out1 = [mul25_out1 as i64];
        let concat22_out1: [i64; 3usize] = [
            &unsqueeze39_out1[..],
            &unsqueeze40_out1[..],
            &unsqueeze41_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape17_out1 = transpose15_out1.reshape(concat22_out1);
        let linear16_out1 = self.linear16.forward(reshape17_out1);
        let add20_out1 = transpose2_out1.add(linear16_out1);
        let layernormalization4_out1 = {
            let dtype = add20_out1.dtype();
            self.layernormalization4
                .forward(add20_out1.cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let linear17_out1 = self.linear17.forward(add19_out1.clone());
        let linear18_out1 = self.linear18.forward(add19_out1);
        let linear19_out1 = self.linear19.forward(layernormalization3_out1.clone());
        let shape52_out1: [i64; 3] = {
            let axes = &linear17_out1.clone().dims()[0..3];
            let mut output = [0i64; 3];
            for i in 0..3 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let gather46_out1 = shape52_out1[0] as i64;
        let gather47_out1 = shape52_out1[1] as i64;
        let gather48_out1 = shape52_out1[2] as i64;
        let constant307_out1 = 8i64;
        let div18_out1 = gather48_out1 / constant307_out1;
        let unsqueeze42_out1 = [gather46_out1 as i64];
        let unsqueeze43_out1 = [gather47_out1 as i64];
        let unsqueeze44_out1 = [div18_out1 as i64];
        let constant310_out1: [i64; 1] = [8i64];
        let concat23_out1: [i64; 4usize] = [
            &unsqueeze42_out1[..],
            &unsqueeze43_out1[..],
            &constant310_out1[..],
            &unsqueeze44_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape18_out1 = linear17_out1.reshape(concat23_out1);
        let transpose16_out1 = reshape18_out1.permute([0, 2, 1, 3]);
        let shape55_out1: [i64; 3] = {
            let axes = &linear18_out1.clone().dims()[0..3];
            let mut output = [0i64; 3];
            for i in 0..3 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let gather49_out1 = shape55_out1[0] as i64;
        let gather50_out1 = shape55_out1[1] as i64;
        let gather51_out1 = shape55_out1[2] as i64;
        let constant315_out1 = 8i64;
        let div19_out1 = gather51_out1 / constant315_out1;
        let unsqueeze45_out1 = [gather49_out1 as i64];
        let unsqueeze46_out1 = [gather50_out1 as i64];
        let unsqueeze47_out1 = [div19_out1 as i64];
        let constant318_out1: [i64; 1] = [8i64];
        let concat24_out1: [i64; 4usize] = [
            &unsqueeze45_out1[..],
            &unsqueeze46_out1[..],
            &constant318_out1[..],
            &unsqueeze47_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape19_out1 = linear18_out1.reshape(concat24_out1);
        let shape58_out1: [i64; 3] = {
            let axes = &linear19_out1.clone().dims()[0..3];
            let mut output = [0i64; 3];
            for i in 0..3 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let gather52_out1 = shape58_out1[0] as i64;
        let gather53_out1 = shape58_out1[1] as i64;
        let gather54_out1 = shape58_out1[2] as i64;
        let constant323_out1 = 8i64;
        let div20_out1 = gather54_out1 / constant323_out1;
        let unsqueeze48_out1 = [gather52_out1 as i64];
        let unsqueeze49_out1 = [gather53_out1 as i64];
        let unsqueeze50_out1 = [div20_out1 as i64];
        let constant326_out1: [i64; 1] = [8i64];
        let concat25_out1: [i64; 4usize] = [
            &unsqueeze48_out1[..],
            &unsqueeze49_out1[..],
            &constant326_out1[..],
            &unsqueeze50_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape20_out1 = linear19_out1.reshape(concat25_out1);
        let transpose17_out1 = reshape20_out1.permute([0, 2, 1, 3]);
        let transpose18_out1 = reshape19_out1.permute([0, 2, 3, 1]);
        let matmul26_out1 = transpose16_out1.matmul(transpose18_out1);
        let constant328_out1 = self.constant328.val();
        let div21_out1 = matmul26_out1
            .div((constant328_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let softmax4_out1 = burn::tensor::activation::softmax(div21_out1, 3);
        let matmul27_out1 = softmax4_out1.matmul(transpose17_out1);
        let shape61_out1: [i64; 4] = {
            let axes = &matmul27_out1.clone().dims()[0..4];
            let mut output = [0i64; 4];
            for i in 0..4 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let gather55_out1 = shape61_out1[0] as i64;
        let gather56_out1 = shape61_out1[1] as i64;
        let gather57_out1 = shape61_out1[2] as i64;
        let gather58_out1 = shape61_out1[3] as i64;
        let transpose19_out1 = matmul27_out1.permute([0, 2, 1, 3]);
        let mul26_out1 = gather56_out1 * gather58_out1;
        let unsqueeze51_out1 = [gather55_out1 as i64];
        let unsqueeze52_out1 = [gather57_out1 as i64];
        let unsqueeze53_out1 = [mul26_out1 as i64];
        let concat26_out1: [i64; 3usize] = [
            &unsqueeze51_out1[..],
            &unsqueeze52_out1[..],
            &unsqueeze53_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape21_out1 = transpose19_out1.reshape(concat26_out1);
        let linear20_out1 = self.linear20.forward(reshape21_out1);
        let add21_out1 = layernormalization3_out1.add(linear20_out1);
        (add21_out1, layernormalization4_out1)
    }
}
#[derive(Module, Debug)]
pub struct Submodule5 {
    layernormalization5: LayerNorm,
    linear21: Linear,
    linear22: Linear,
    linear23: Linear,
    constant360: burn::module::Param<Tensor<1>>,
    linear24: Linear,
    layernormalization6: LayerNorm,
    linear25: Linear,
    linear26: Linear,
    layernormalization7: LayerNorm,
    linear27: Linear,
    linear28: Linear,
    linear29: Linear,
    constant392: burn::module::Param<Tensor<1>>,
    linear30: Linear,
    layernormalization8: LayerNorm,
    linear31: Linear,
    linear32: Linear,
    linear33: Linear,
    constant424: burn::module::Param<Tensor<1>>,
    linear34: Linear,
    layernormalization9: LayerNorm,
    #[module(skip)]
    device: Device,
}
impl Submodule5 {
    #[allow(unused_variables)]
    pub fn new(device: &Device) -> Self {
        let layernormalization5 = LayerNormConfig::new(256)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(true)
            .init(device);
        let linear21 = LinearConfig::new(256, 128).with_bias(true).init(device);
        let linear22 = LinearConfig::new(256, 128).with_bias(true).init(device);
        let linear23 = LinearConfig::new(256, 128).with_bias(true).init(device);
        let constant360: burn::module::Param<Tensor<1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<
                1,
            >::from_data(
                burn::tensor::TensorData::from([4f64]),
                (device, burn::tensor::DType::F32),
            ),
            device.clone(),
            false,
            [1].into(),
        );
        let linear24 = LinearConfig::new(128, 256).with_bias(true).init(device);
        let layernormalization6 = LayerNormConfig::new(256)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(true)
            .init(device);
        let linear25 = LinearConfig::new(256, 2048).with_bias(true).init(device);
        let linear26 = LinearConfig::new(2048, 256).with_bias(true).init(device);
        let layernormalization7 = LayerNormConfig::new(256)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(true)
            .init(device);
        let linear27 = LinearConfig::new(256, 128).with_bias(true).init(device);
        let linear28 = LinearConfig::new(256, 128).with_bias(true).init(device);
        let linear29 = LinearConfig::new(256, 128).with_bias(true).init(device);
        let constant392: burn::module::Param<Tensor<1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<
                1,
            >::from_data(
                burn::tensor::TensorData::from([4f64]),
                (device, burn::tensor::DType::F32),
            ),
            device.clone(),
            false,
            [1].into(),
        );
        let linear30 = LinearConfig::new(128, 256).with_bias(true).init(device);
        let layernormalization8 = LayerNormConfig::new(256)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(true)
            .init(device);
        let linear31 = LinearConfig::new(256, 128).with_bias(true).init(device);
        let linear32 = LinearConfig::new(256, 128).with_bias(true).init(device);
        let linear33 = LinearConfig::new(256, 128).with_bias(true).init(device);
        let constant424: burn::module::Param<Tensor<1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<
                1,
            >::from_data(
                burn::tensor::TensorData::from([4f64]),
                (device, burn::tensor::DType::F32),
            ),
            device.clone(),
            false,
            [1].into(),
        );
        let linear34 = LinearConfig::new(128, 256).with_bias(true).init(device);
        let layernormalization9 = LayerNormConfig::new(256)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(true)
            .init(device);
        Self {
            layernormalization5,
            linear21,
            linear22,
            linear23,
            constant360,
            linear24,
            layernormalization6,
            linear25,
            linear26,
            layernormalization7,
            linear27,
            linear28,
            linear29,
            constant392,
            linear30,
            layernormalization8,
            linear31,
            linear32,
            linear33,
            constant424,
            linear34,
            layernormalization9,
            device: device.clone(),
        }
    }
    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(
        &self,
        add21_out1: Tensor<3>,
        concat4_out1: Tensor<3>,
        layernormalization4_out1: Tensor<3>,
        transpose3_out1: Tensor<3>,
        gather3_out1: i64,
        gather4_out1: i64,
        gather5_out1: i64,
        gather6_out1: i64,
    ) -> (Tensor<4>, Tensor<3>, Tensor<2>) {
        let layernormalization5_out1 = {
            let dtype = add21_out1.dtype();
            self.layernormalization5
                .forward(add21_out1.cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let add22_out1 = layernormalization5_out1.clone().add(concat4_out1.clone());
        let add23_out1 = layernormalization4_out1.clone().add(transpose3_out1.clone());
        let linear21_out1 = self.linear21.forward(add22_out1);
        let linear22_out1 = self.linear22.forward(add23_out1.clone());
        let linear23_out1 = self.linear23.forward(layernormalization4_out1.clone());
        let shape65_out1: [i64; 3] = {
            let axes = &linear21_out1.clone().dims()[0..3];
            let mut output = [0i64; 3];
            for i in 0..3 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let gather59_out1 = shape65_out1[0] as i64;
        let gather60_out1 = shape65_out1[1] as i64;
        let gather61_out1 = shape65_out1[2] as i64;
        let constant339_out1 = 8i64;
        let div22_out1 = gather61_out1 / constant339_out1;
        let unsqueeze54_out1 = [gather59_out1 as i64];
        let unsqueeze55_out1 = [gather60_out1 as i64];
        let unsqueeze56_out1 = [div22_out1 as i64];
        let constant342_out1: [i64; 1] = [8i64];
        let concat27_out1: [i64; 4usize] = [
            &unsqueeze54_out1[..],
            &unsqueeze55_out1[..],
            &constant342_out1[..],
            &unsqueeze56_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape22_out1 = linear21_out1.reshape(concat27_out1);
        let transpose20_out1 = reshape22_out1.permute([0, 2, 1, 3]);
        let shape68_out1: [i64; 3] = {
            let axes = &linear22_out1.clone().dims()[0..3];
            let mut output = [0i64; 3];
            for i in 0..3 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let gather62_out1 = shape68_out1[0] as i64;
        let gather63_out1 = shape68_out1[1] as i64;
        let gather64_out1 = shape68_out1[2] as i64;
        let constant347_out1 = 8i64;
        let div23_out1 = gather64_out1 / constant347_out1;
        let unsqueeze57_out1 = [gather62_out1 as i64];
        let unsqueeze58_out1 = [gather63_out1 as i64];
        let unsqueeze59_out1 = [div23_out1 as i64];
        let constant350_out1: [i64; 1] = [8i64];
        let concat28_out1: [i64; 4usize] = [
            &unsqueeze57_out1[..],
            &unsqueeze58_out1[..],
            &constant350_out1[..],
            &unsqueeze59_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape23_out1 = linear22_out1.reshape(concat28_out1);
        let shape71_out1: [i64; 3] = {
            let axes = &linear23_out1.clone().dims()[0..3];
            let mut output = [0i64; 3];
            for i in 0..3 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let gather65_out1 = shape71_out1[0] as i64;
        let gather66_out1 = shape71_out1[1] as i64;
        let gather67_out1 = shape71_out1[2] as i64;
        let constant355_out1 = 8i64;
        let div24_out1 = gather67_out1 / constant355_out1;
        let unsqueeze60_out1 = [gather65_out1 as i64];
        let unsqueeze61_out1 = [gather66_out1 as i64];
        let unsqueeze62_out1 = [div24_out1 as i64];
        let constant358_out1: [i64; 1] = [8i64];
        let concat29_out1: [i64; 4usize] = [
            &unsqueeze60_out1[..],
            &unsqueeze61_out1[..],
            &constant358_out1[..],
            &unsqueeze62_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape24_out1 = linear23_out1.reshape(concat29_out1);
        let transpose21_out1 = reshape24_out1.permute([0, 2, 1, 3]);
        let transpose22_out1 = reshape23_out1.permute([0, 2, 3, 1]);
        let matmul32_out1 = transpose20_out1.matmul(transpose22_out1);
        let constant360_out1 = self.constant360.val();
        let div25_out1 = matmul32_out1
            .div((constant360_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let softmax5_out1 = burn::tensor::activation::softmax(div25_out1, 3);
        let matmul33_out1 = softmax5_out1.matmul(transpose21_out1);
        let shape74_out1: [i64; 4] = {
            let axes = &matmul33_out1.clone().dims()[0..4];
            let mut output = [0i64; 4];
            for i in 0..4 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let gather68_out1 = shape74_out1[0] as i64;
        let gather69_out1 = shape74_out1[1] as i64;
        let gather70_out1 = shape74_out1[2] as i64;
        let gather71_out1 = shape74_out1[3] as i64;
        let transpose23_out1 = matmul33_out1.permute([0, 2, 1, 3]);
        let mul27_out1 = gather69_out1 * gather71_out1;
        let unsqueeze63_out1 = [gather68_out1 as i64];
        let unsqueeze64_out1 = [gather70_out1 as i64];
        let unsqueeze65_out1 = [mul27_out1 as i64];
        let concat30_out1: [i64; 3usize] = [
            &unsqueeze63_out1[..],
            &unsqueeze64_out1[..],
            &unsqueeze65_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape25_out1 = transpose23_out1.reshape(concat30_out1);
        let linear24_out1 = self.linear24.forward(reshape25_out1);
        let add24_out1 = layernormalization5_out1.add(linear24_out1);
        let layernormalization6_out1 = {
            let dtype = add24_out1.dtype();
            self.layernormalization6
                .forward(add24_out1.cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let linear25_out1 = self.linear25.forward(layernormalization6_out1.clone());
        let relu2_out1 = burn::tensor::activation::relu(linear25_out1);
        let linear26_out1 = self.linear26.forward(relu2_out1);
        let add25_out1 = layernormalization6_out1.add(linear26_out1);
        let layernormalization7_out1 = {
            let dtype = add25_out1.dtype();
            self.layernormalization7
                .forward(add25_out1.cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let add26_out1 = layernormalization7_out1.clone().add(concat4_out1);
        let linear27_out1 = self.linear27.forward(add23_out1);
        let linear28_out1 = self.linear28.forward(add26_out1.clone());
        let linear29_out1 = self.linear29.forward(layernormalization7_out1.clone());
        let shape78_out1: [i64; 3] = {
            let axes = &linear27_out1.clone().dims()[0..3];
            let mut output = [0i64; 3];
            for i in 0..3 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let gather72_out1 = shape78_out1[0] as i64;
        let gather73_out1 = shape78_out1[1] as i64;
        let gather74_out1 = shape78_out1[2] as i64;
        let constant371_out1 = 8i64;
        let div26_out1 = gather74_out1 / constant371_out1;
        let unsqueeze66_out1 = [gather72_out1 as i64];
        let unsqueeze67_out1 = [gather73_out1 as i64];
        let unsqueeze68_out1 = [div26_out1 as i64];
        let constant374_out1: [i64; 1] = [8i64];
        let concat31_out1: [i64; 4usize] = [
            &unsqueeze66_out1[..],
            &unsqueeze67_out1[..],
            &constant374_out1[..],
            &unsqueeze68_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape26_out1 = linear27_out1.reshape(concat31_out1);
        let transpose24_out1 = reshape26_out1.permute([0, 2, 1, 3]);
        let shape81_out1: [i64; 3] = {
            let axes = &linear28_out1.clone().dims()[0..3];
            let mut output = [0i64; 3];
            for i in 0..3 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let gather75_out1 = shape81_out1[0] as i64;
        let gather76_out1 = shape81_out1[1] as i64;
        let gather77_out1 = shape81_out1[2] as i64;
        let constant379_out1 = 8i64;
        let div27_out1 = gather77_out1 / constant379_out1;
        let unsqueeze69_out1 = [gather75_out1 as i64];
        let unsqueeze70_out1 = [gather76_out1 as i64];
        let unsqueeze71_out1 = [div27_out1 as i64];
        let constant382_out1: [i64; 1] = [8i64];
        let concat32_out1: [i64; 4usize] = [
            &unsqueeze69_out1[..],
            &unsqueeze70_out1[..],
            &constant382_out1[..],
            &unsqueeze71_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape27_out1 = linear28_out1.reshape(concat32_out1);
        let shape84_out1: [i64; 3] = {
            let axes = &linear29_out1.clone().dims()[0..3];
            let mut output = [0i64; 3];
            for i in 0..3 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let gather78_out1 = shape84_out1[0] as i64;
        let gather79_out1 = shape84_out1[1] as i64;
        let gather80_out1 = shape84_out1[2] as i64;
        let constant387_out1 = 8i64;
        let div28_out1 = gather80_out1 / constant387_out1;
        let unsqueeze72_out1 = [gather78_out1 as i64];
        let unsqueeze73_out1 = [gather79_out1 as i64];
        let unsqueeze74_out1 = [div28_out1 as i64];
        let constant390_out1: [i64; 1] = [8i64];
        let concat33_out1: [i64; 4usize] = [
            &unsqueeze72_out1[..],
            &unsqueeze73_out1[..],
            &constant390_out1[..],
            &unsqueeze74_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape28_out1 = linear29_out1.reshape(concat33_out1);
        let transpose25_out1 = reshape28_out1.permute([0, 2, 1, 3]);
        let transpose26_out1 = reshape27_out1.permute([0, 2, 3, 1]);
        let matmul40_out1 = transpose24_out1.matmul(transpose26_out1);
        let constant392_out1 = self.constant392.val();
        let div29_out1 = matmul40_out1
            .div((constant392_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let softmax6_out1 = burn::tensor::activation::softmax(div29_out1, 3);
        let matmul41_out1 = softmax6_out1.matmul(transpose25_out1);
        let shape87_out1: [i64; 4] = {
            let axes = &matmul41_out1.clone().dims()[0..4];
            let mut output = [0i64; 4];
            for i in 0..4 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let gather81_out1 = shape87_out1[0] as i64;
        let gather82_out1 = shape87_out1[1] as i64;
        let gather83_out1 = shape87_out1[2] as i64;
        let gather84_out1 = shape87_out1[3] as i64;
        let transpose27_out1 = matmul41_out1.permute([0, 2, 1, 3]);
        let mul28_out1 = gather82_out1 * gather84_out1;
        let unsqueeze75_out1 = [gather81_out1 as i64];
        let unsqueeze76_out1 = [gather83_out1 as i64];
        let unsqueeze77_out1 = [mul28_out1 as i64];
        let concat34_out1: [i64; 3usize] = [
            &unsqueeze75_out1[..],
            &unsqueeze76_out1[..],
            &unsqueeze77_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape29_out1 = transpose27_out1.reshape(concat34_out1);
        let linear30_out1 = self.linear30.forward(reshape29_out1);
        let add27_out1 = layernormalization4_out1.add(linear30_out1);
        let layernormalization8_out1 = {
            let dtype = add27_out1.dtype();
            self.layernormalization8
                .forward(add27_out1.cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let add28_out1 = layernormalization8_out1.clone().add(transpose3_out1);
        let linear31_out1 = self.linear31.forward(add26_out1);
        let linear32_out1 = self.linear32.forward(add28_out1);
        let linear33_out1 = self.linear33.forward(layernormalization8_out1.clone());
        let shape91_out1: [i64; 3] = {
            let axes = &linear31_out1.clone().dims()[0..3];
            let mut output = [0i64; 3];
            for i in 0..3 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let gather85_out1 = shape91_out1[0] as i64;
        let gather86_out1 = shape91_out1[1] as i64;
        let gather87_out1 = shape91_out1[2] as i64;
        let constant403_out1 = 8i64;
        let div30_out1 = gather87_out1 / constant403_out1;
        let unsqueeze78_out1 = [gather85_out1 as i64];
        let unsqueeze79_out1 = [gather86_out1 as i64];
        let unsqueeze80_out1 = [div30_out1 as i64];
        let constant406_out1: [i64; 1] = [8i64];
        let concat35_out1: [i64; 4usize] = [
            &unsqueeze78_out1[..],
            &unsqueeze79_out1[..],
            &constant406_out1[..],
            &unsqueeze80_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape30_out1 = linear31_out1.reshape(concat35_out1);
        let transpose28_out1 = reshape30_out1.permute([0, 2, 1, 3]);
        let shape94_out1: [i64; 3] = {
            let axes = &linear32_out1.clone().dims()[0..3];
            let mut output = [0i64; 3];
            for i in 0..3 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let gather88_out1 = shape94_out1[0] as i64;
        let gather89_out1 = shape94_out1[1] as i64;
        let gather90_out1 = shape94_out1[2] as i64;
        let constant411_out1 = 8i64;
        let div31_out1 = gather90_out1 / constant411_out1;
        let unsqueeze81_out1 = [gather88_out1 as i64];
        let unsqueeze82_out1 = [gather89_out1 as i64];
        let unsqueeze83_out1 = [div31_out1 as i64];
        let constant414_out1: [i64; 1] = [8i64];
        let concat36_out1: [i64; 4usize] = [
            &unsqueeze81_out1[..],
            &unsqueeze82_out1[..],
            &constant414_out1[..],
            &unsqueeze83_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape31_out1 = linear32_out1.reshape(concat36_out1);
        let shape97_out1: [i64; 3] = {
            let axes = &linear33_out1.clone().dims()[0..3];
            let mut output = [0i64; 3];
            for i in 0..3 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let gather91_out1 = shape97_out1[0] as i64;
        let gather92_out1 = shape97_out1[1] as i64;
        let gather93_out1 = shape97_out1[2] as i64;
        let constant419_out1 = 8i64;
        let div32_out1 = gather93_out1 / constant419_out1;
        let unsqueeze84_out1 = [gather91_out1 as i64];
        let unsqueeze85_out1 = [gather92_out1 as i64];
        let unsqueeze86_out1 = [div32_out1 as i64];
        let constant422_out1: [i64; 1] = [8i64];
        let concat37_out1: [i64; 4usize] = [
            &unsqueeze84_out1[..],
            &unsqueeze85_out1[..],
            &constant422_out1[..],
            &unsqueeze86_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape32_out1 = linear33_out1.reshape(concat37_out1);
        let transpose29_out1 = reshape32_out1.permute([0, 2, 1, 3]);
        let transpose30_out1 = reshape31_out1.permute([0, 2, 3, 1]);
        let matmul46_out1 = transpose28_out1.matmul(transpose30_out1);
        let constant424_out1 = self.constant424.val();
        let div33_out1 = matmul46_out1
            .div((constant424_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let softmax7_out1 = burn::tensor::activation::softmax(div33_out1, 3);
        let matmul47_out1 = softmax7_out1.matmul(transpose29_out1);
        let shape100_out1: [i64; 4] = {
            let axes = &matmul47_out1.clone().dims()[0..4];
            let mut output = [0i64; 4];
            for i in 0..4 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let gather94_out1 = shape100_out1[0] as i64;
        let gather95_out1 = shape100_out1[1] as i64;
        let gather96_out1 = shape100_out1[2] as i64;
        let gather97_out1 = shape100_out1[3] as i64;
        let transpose31_out1 = matmul47_out1.permute([0, 2, 1, 3]);
        let mul29_out1 = gather95_out1 * gather97_out1;
        let unsqueeze87_out1 = [gather94_out1 as i64];
        let unsqueeze88_out1 = [gather96_out1 as i64];
        let unsqueeze89_out1 = [mul29_out1 as i64];
        let concat38_out1: [i64; 3usize] = [
            &unsqueeze87_out1[..],
            &unsqueeze88_out1[..],
            &unsqueeze89_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape33_out1 = transpose31_out1.reshape(concat38_out1);
        let linear34_out1 = self.linear34.forward(reshape33_out1);
        let add29_out1 = layernormalization7_out1.add(linear34_out1);
        let layernormalization9_out1 = {
            let dtype = add29_out1.dtype();
            self.layernormalization9
                .forward(add29_out1.cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let gather98_out1 = {
            let sliced = layernormalization9_out1.clone().slice(s![.., 0, ..]);
            sliced.squeeze_dim::<2usize>(1)
        };
        let slice3_out1 = layernormalization9_out1.slice(s![.., 1..5, ..]);
        let transpose32_out1 = layernormalization8_out1.permute([0, 2, 1]);
        let unsqueeze90_out1 = [gather3_out1 as i64];
        let unsqueeze91_out1 = [gather4_out1 as i64];
        let unsqueeze92_out1 = [gather5_out1 as i64];
        let unsqueeze93_out1 = [gather6_out1 as i64];
        let concat39_out1: [i64; 4usize] = [
            &unsqueeze90_out1[..],
            &unsqueeze91_out1[..],
            &unsqueeze92_out1[..],
            &unsqueeze93_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape34_out1 = transpose32_out1.reshape(concat39_out1);
        (reshape34_out1, slice3_out1, gather98_out1)
    }
}
#[derive(Module, Debug)]
pub struct Submodule6 {
    convtranspose2d1: ConvTranspose2d,
    constant440: burn::module::Param<Tensor<1>>,
    constant441: burn::module::Param<Tensor<1>>,
    constant135: burn::module::Param<Tensor<3>>,
    constant136: burn::module::Param<Tensor<3>>,
    constant442: burn::module::Param<Tensor<1>>,
    constant443: burn::module::Param<Tensor<1>>,
    constant444: burn::module::Param<Tensor<1>>,
    convtranspose2d2: ConvTranspose2d,
    constant445: burn::module::Param<Tensor<1>>,
    constant446: burn::module::Param<Tensor<1>>,
    constant447: burn::module::Param<Tensor<1>>,
    linear35: Linear,
    linear36: Linear,
    linear37: Linear,
    linear38: Linear,
    linear39: Linear,
    linear40: Linear,
    linear41: Linear,
    linear42: Linear,
    linear43: Linear,
    linear44: Linear,
    linear45: Linear,
    linear46: Linear,
    linear47: Linear,
    linear48: Linear,
    linear49: Linear,
    #[module(skip)]
    device: Device,
}
impl Submodule6 {
    #[allow(unused_variables)]
    pub fn new(device: &Device) -> Self {
        let convtranspose2d1 = ConvTranspose2dConfig::new([256, 64], [2, 2])
            .with_stride([2, 2])
            .with_padding([0, 0])
            .with_padding_out([0, 0])
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let constant440: burn::module::Param<Tensor<1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<
                1,
            >::from_data(
                burn::tensor::TensorData::from([2f64]),
                (device, burn::tensor::DType::F32),
            ),
            device.clone(),
            false,
            [1].into(),
        );
        let constant441: burn::module::Param<Tensor<1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<
                1,
            >::from_data(
                burn::tensor::TensorData::from([0.0000009999999974752427f64]),
                (device, burn::tensor::DType::F32),
            ),
            device.clone(),
            false,
            [1].into(),
        );
        let constant135: burn::module::Param<Tensor<3>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<
                3,
            >::zeros([64, 1, 1], (device, burn::tensor::DType::F32)),
            device.clone(),
            false,
            [64, 1, 1].into(),
        );
        let constant136: burn::module::Param<Tensor<3>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<
                3,
            >::zeros([64, 1, 1], (device, burn::tensor::DType::F32)),
            device.clone(),
            false,
            [64, 1, 1].into(),
        );
        let constant442: burn::module::Param<Tensor<1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<
                1,
            >::from_data(
                burn::tensor::TensorData::from([1.4142135381698608f64]),
                (device, burn::tensor::DType::F32),
            ),
            device.clone(),
            false,
            [1].into(),
        );
        let constant443: burn::module::Param<Tensor<1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<
                1,
            >::from_data(
                burn::tensor::TensorData::from([1f64]),
                (device, burn::tensor::DType::F32),
            ),
            device.clone(),
            false,
            [1].into(),
        );
        let constant444: burn::module::Param<Tensor<1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<
                1,
            >::from_data(
                burn::tensor::TensorData::from([0.5f64]),
                (device, burn::tensor::DType::F32),
            ),
            device.clone(),
            false,
            [1].into(),
        );
        let convtranspose2d2 = ConvTranspose2dConfig::new([64, 32], [2, 2])
            .with_stride([2, 2])
            .with_padding([0, 0])
            .with_padding_out([0, 0])
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let constant445: burn::module::Param<Tensor<1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<
                1,
            >::from_data(
                burn::tensor::TensorData::from([1.4142135381698608f64]),
                (device, burn::tensor::DType::F32),
            ),
            device.clone(),
            false,
            [1].into(),
        );
        let constant446: burn::module::Param<Tensor<1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<
                1,
            >::from_data(
                burn::tensor::TensorData::from([1f64]),
                (device, burn::tensor::DType::F32),
            ),
            device.clone(),
            false,
            [1].into(),
        );
        let constant447: burn::module::Param<Tensor<1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<
                1,
            >::from_data(
                burn::tensor::TensorData::from([0.5f64]),
                (device, burn::tensor::DType::F32),
            ),
            device.clone(),
            false,
            [1].into(),
        );
        let linear35 = LinearConfig::new(256, 256).with_bias(true).init(device);
        let linear36 = LinearConfig::new(256, 256).with_bias(true).init(device);
        let linear37 = LinearConfig::new(256, 32).with_bias(true).init(device);
        let linear38 = LinearConfig::new(256, 256).with_bias(true).init(device);
        let linear39 = LinearConfig::new(256, 256).with_bias(true).init(device);
        let linear40 = LinearConfig::new(256, 32).with_bias(true).init(device);
        let linear41 = LinearConfig::new(256, 256).with_bias(true).init(device);
        let linear42 = LinearConfig::new(256, 256).with_bias(true).init(device);
        let linear43 = LinearConfig::new(256, 32).with_bias(true).init(device);
        let linear44 = LinearConfig::new(256, 256).with_bias(true).init(device);
        let linear45 = LinearConfig::new(256, 256).with_bias(true).init(device);
        let linear46 = LinearConfig::new(256, 32).with_bias(true).init(device);
        let linear47 = LinearConfig::new(256, 256).with_bias(true).init(device);
        let linear48 = LinearConfig::new(256, 256).with_bias(true).init(device);
        let linear49 = LinearConfig::new(256, 4).with_bias(true).init(device);
        Self {
            convtranspose2d1,
            constant440,
            constant441,
            constant135,
            constant136,
            constant442,
            constant443,
            constant444,
            convtranspose2d2,
            constant445,
            constant446,
            constant447,
            linear35,
            linear36,
            linear37,
            linear38,
            linear39,
            linear40,
            linear41,
            linear42,
            linear43,
            linear44,
            linear45,
            linear46,
            linear47,
            linear48,
            linear49,
            device: device.clone(),
        }
    }
    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(
        &self,
        reshape34_out1: Tensor<4>,
        slice3_out1: Tensor<3>,
        gather98_out1: Tensor<2>,
    ) -> (Tensor<2>, Tensor<4>) {
        let convtranspose2d1_out1 = self.convtranspose2d1.forward(reshape34_out1);
        let reducemean5_out1 = { convtranspose2d1_out1.clone().mean_dim(1usize) };
        let sub5_out1 = convtranspose2d1_out1.sub(reducemean5_out1);
        let constant440_out1 = self.constant440.val();
        let pow3_out1 = sub5_out1
            .clone()
            .powf((constant440_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let reducemean6_out1 = { pow3_out1.mean_dim(1usize) };
        let constant441_out1 = self.constant441.val();
        let add30_out1 = reducemean6_out1
            .add((constant441_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let sqrt3_out1 = add30_out1.sqrt();
        let div34_out1 = sub5_out1.div(sqrt3_out1);
        let constant135_out1 = self.constant135.val();
        let mul30_out1 = (constant135_out1).unsqueeze_dims(&[0isize]).mul(div34_out1);
        let constant136_out1 = self.constant136.val();
        let add31_out1 = mul30_out1.add((constant136_out1).unsqueeze_dims(&[0isize]));
        let constant442_out1 = self.constant442.val();
        let div35_out1 = add31_out1
            .clone()
            .div((constant442_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let erf3_out1 = div35_out1.erf();
        let constant443_out1 = self.constant443.val();
        let add32_out1 = erf3_out1
            .add((constant443_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let mul31_out1 = add31_out1.mul(add32_out1);
        let constant444_out1 = self.constant444.val();
        let mul32_out1 = mul31_out1
            .mul((constant444_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let convtranspose2d2_out1 = self.convtranspose2d2.forward(mul32_out1);
        let constant445_out1 = self.constant445.val();
        let div36_out1 = convtranspose2d2_out1
            .clone()
            .div((constant445_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let erf4_out1 = div36_out1.erf();
        let constant446_out1 = self.constant446.val();
        let add33_out1 = erf4_out1
            .add((constant446_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let mul33_out1 = convtranspose2d2_out1.mul(add33_out1);
        let constant447_out1 = self.constant447.val();
        let mul34_out1 = mul33_out1
            .mul((constant447_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let gather99_out1 = {
            let sliced = slice3_out1.clone().slice(s![.., 0, ..]);
            sliced.squeeze_dim::<2usize>(1)
        };
        let linear35_out1 = self.linear35.forward(gather99_out1);
        let relu3_out1 = burn::tensor::activation::relu(linear35_out1);
        let linear36_out1 = self.linear36.forward(relu3_out1);
        let relu4_out1 = burn::tensor::activation::relu(linear36_out1);
        let linear37_out1 = self.linear37.forward(relu4_out1);
        let gather100_out1 = {
            let sliced = slice3_out1.clone().slice(s![.., 1, ..]);
            sliced.squeeze_dim::<2usize>(1)
        };
        let linear38_out1 = self.linear38.forward(gather100_out1);
        let relu5_out1 = burn::tensor::activation::relu(linear38_out1);
        let linear39_out1 = self.linear39.forward(relu5_out1);
        let relu6_out1 = burn::tensor::activation::relu(linear39_out1);
        let linear40_out1 = self.linear40.forward(relu6_out1);
        let gather101_out1 = {
            let sliced = slice3_out1.clone().slice(s![.., 2, ..]);
            sliced.squeeze_dim::<2usize>(1)
        };
        let linear41_out1 = self.linear41.forward(gather101_out1);
        let relu7_out1 = burn::tensor::activation::relu(linear41_out1);
        let linear42_out1 = self.linear42.forward(relu7_out1);
        let relu8_out1 = burn::tensor::activation::relu(linear42_out1);
        let linear43_out1 = self.linear43.forward(relu8_out1);
        let gather102_out1 = {
            let sliced = slice3_out1.slice(s![.., 3, ..]);
            sliced.squeeze_dim::<2usize>(1)
        };
        let linear44_out1 = self.linear44.forward(gather102_out1);
        let relu9_out1 = burn::tensor::activation::relu(linear44_out1);
        let linear45_out1 = self.linear45.forward(relu9_out1);
        let relu10_out1 = burn::tensor::activation::relu(linear45_out1);
        let linear46_out1 = self.linear46.forward(relu10_out1);
        let unsqueeze94_out1: Tensor<3> = linear37_out1.unsqueeze_dims::<3>(&[1]);
        let unsqueeze95_out1: Tensor<3> = linear40_out1.unsqueeze_dims::<3>(&[1]);
        let unsqueeze96_out1: Tensor<3> = linear43_out1.unsqueeze_dims::<3>(&[1]);
        let unsqueeze97_out1: Tensor<3> = linear46_out1.unsqueeze_dims::<3>(&[1]);
        let concat40_out1 = burn::tensor::Tensor::cat(
            [unsqueeze94_out1, unsqueeze95_out1, unsqueeze96_out1, unsqueeze97_out1]
                .into(),
            1,
        );
        let shape104_out1: [i64; 4] = {
            let axes = &mul34_out1.clone().dims()[0..4];
            let mut output = [0i64; 4];
            for i in 0..4 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let gather103_out1 = shape104_out1[0] as i64;
        let gather104_out1 = shape104_out1[1] as i64;
        let gather105_out1 = shape104_out1[2] as i64;
        let gather106_out1 = shape104_out1[3] as i64;
        let mul35_out1 = gather105_out1 * gather106_out1;
        let unsqueeze98_out1 = [gather103_out1 as i64];
        let unsqueeze99_out1 = [gather104_out1 as i64];
        let unsqueeze100_out1 = [mul35_out1 as i64];
        let concat41_out1: [i64; 3usize] = [
            &unsqueeze98_out1[..],
            &unsqueeze99_out1[..],
            &unsqueeze100_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape35_out1 = mul34_out1.reshape(concat41_out1);
        let matmul49_out1 = concat40_out1.matmul(reshape35_out1);
        let unsqueeze101_out1 = [gather103_out1 as i64];
        let unsqueeze102_out1 = [gather105_out1 as i64];
        let unsqueeze103_out1 = [gather106_out1 as i64];
        let constant460_out1: [i64; 1] = [-1i64];
        let concat42_out1: [i64; 4usize] = [
            &unsqueeze101_out1[..],
            &constant460_out1[..],
            &unsqueeze102_out1[..],
            &unsqueeze103_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape36_out1 = matmul49_out1.reshape(concat42_out1);
        let linear47_out1 = self.linear47.forward(gather98_out1);
        let relu11_out1 = burn::tensor::activation::relu(linear47_out1);
        let linear48_out1 = self.linear48.forward(relu11_out1);
        let relu12_out1 = burn::tensor::activation::relu(linear48_out1);
        let linear49_out1 = self.linear49.forward(relu12_out1);
        (linear49_out1, reshape36_out1)
    }
}

#[derive(Module, Debug)]
pub struct Model {
    submodule1: Submodule1,
    submodule2: Submodule2,
    submodule3: Submodule3,
    submodule4: Submodule4,
    submodule5: Submodule5,
    submodule6: Submodule6,
    #[module(skip)]
    device: Device,
}

impl Model {
    #[allow(unused_variables)]
    pub fn new(device: &Device) -> Self {
        let submodule1 = Submodule1::new(device);
        let submodule2 = Submodule2::new(device);
        let submodule3 = Submodule3::new(device);
        let submodule4 = Submodule4::new(device);
        let submodule5 = Submodule5::new(device);
        let submodule6 = Submodule6::new(device);
        Self {
            submodule1,
            submodule2,
            submodule3,
            submodule4,
            submodule5,
            submodule6,
            device: device.clone(),
        }
    }

    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(
        &self,
        image_embeddings: Tensor<4>,
        point_coords: Tensor<3>,
        point_labels: Tensor<2>,
        mask_input: Tensor<4>,
        has_mask_input: Tensor<1>,
    ) -> (Tensor<2>, Tensor<4>) {
        let (add6_out1, add13_out1) = self
            .submodule1
            .forward(point_coords, point_labels, mask_input, has_mask_input);
        let (expand6_out1, add14_out1, concat4_out1) = self
            .submodule2
            .forward(add6_out1, image_embeddings, add13_out1);
        let (
            matmul7_out1,
            transpose2_out1,
            transpose3_out1,
            gather3_out1,
            gather4_out1,
            gather5_out1,
            gather6_out1,
        ) = self.submodule3.forward(expand6_out1, add14_out1, concat4_out1.clone());
        let (add21_out1, layernormalization4_out1) = self
            .submodule4
            .forward(
                matmul7_out1,
                concat4_out1.clone(),
                transpose2_out1,
                transpose3_out1.clone(),
            );
        let (reshape34_out1, slice3_out1, gather98_out1) = self
            .submodule5
            .forward(
                add21_out1,
                concat4_out1,
                layernormalization4_out1,
                transpose3_out1,
                gather3_out1,
                gather4_out1,
                gather5_out1,
                gather6_out1,
            );
        let (linear49_out1, reshape36_out1) = self
            .submodule6
            .forward(reshape34_out1, slice3_out1, gather98_out1);
        (linear49_out1, reshape36_out1)
    }
}
