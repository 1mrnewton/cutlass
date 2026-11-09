// Generated from ONNX "../ml-bench/models/mobile_sam_image_encoder.onnx" by burn-onnx
extern crate alloc;
use burn::prelude::*;
use burn::nn::LayerNorm;
use burn::nn::LayerNormConfig;
use burn::nn::Linear;
use burn::nn::LinearConfig;
use burn::nn::PaddingConfig2d;
use burn::nn::conv::Conv2d;
use burn::nn::conv::Conv2dConfig;
use burn::tensor::TensorData;


#[derive(Module, Debug)]
pub struct Submodule1 {
    constant181: burn::module::Param<Tensor<1>>,
    constant182: burn::module::Param<Tensor<1>>,
    conv2d1: Conv2d,
    constant204: burn::module::Param<Tensor<1>>,
    constant205: burn::module::Param<Tensor<1>>,
    constant206: burn::module::Param<Tensor<1>>,
    conv2d2: Conv2d,
    conv2d3: Conv2d,
    constant207: burn::module::Param<Tensor<1>>,
    constant208: burn::module::Param<Tensor<1>>,
    constant209: burn::module::Param<Tensor<1>>,
    conv2d4: Conv2d,
    constant210: burn::module::Param<Tensor<1>>,
    constant211: burn::module::Param<Tensor<1>>,
    constant212: burn::module::Param<Tensor<1>>,
    conv2d5: Conv2d,
    #[module(skip)]
    device: Device,
}
impl Submodule1 {
    #[allow(unused_variables)]
    pub fn new(device: &Device) -> Self {
        let constant181: burn::module::Param<Tensor<1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<
                1,
            >::zeros([3], (device, burn::tensor::DType::F32)),
            device.clone(),
            false,
            [3].into(),
        );
        let constant182: burn::module::Param<Tensor<1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<
                1,
            >::zeros([3], (device, burn::tensor::DType::F32)),
            device.clone(),
            false,
            [3].into(),
        );
        let conv2d1 = Conv2dConfig::new([3, 32], [3, 3])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let constant204: burn::module::Param<Tensor<1>> = burn::module::Param::uninitialized(
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
        let constant205: burn::module::Param<Tensor<1>> = burn::module::Param::uninitialized(
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
        let constant206: burn::module::Param<Tensor<1>> = burn::module::Param::uninitialized(
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
        let conv2d2 = Conv2dConfig::new([32, 64], [3, 3])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d3 = Conv2dConfig::new([64, 256], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let constant207: burn::module::Param<Tensor<1>> = burn::module::Param::uninitialized(
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
        let constant208: burn::module::Param<Tensor<1>> = burn::module::Param::uninitialized(
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
        let constant209: burn::module::Param<Tensor<1>> = burn::module::Param::uninitialized(
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
        let conv2d4 = Conv2dConfig::new([256, 256], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(256)
            .with_bias(true)
            .init(device);
        let constant210: burn::module::Param<Tensor<1>> = burn::module::Param::uninitialized(
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
        let constant211: burn::module::Param<Tensor<1>> = burn::module::Param::uninitialized(
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
        let constant212: burn::module::Param<Tensor<1>> = burn::module::Param::uninitialized(
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
        let conv2d5 = Conv2dConfig::new([256, 64], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        Self {
            constant181,
            constant182,
            conv2d1,
            constant204,
            constant205,
            constant206,
            conv2d2,
            conv2d3,
            constant207,
            constant208,
            constant209,
            conv2d4,
            constant210,
            constant211,
            constant212,
            conv2d5,
            device: device.clone(),
        }
    }
    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(&self, input_image: Tensor<3>) -> Tensor<4> {
        let constant181_out1 = self.constant181.val();
        let sub1_out1 = input_image
            .sub((constant181_out1).unsqueeze_dims(&[0isize, 1isize]));
        let constant182_out1 = self.constant182.val();
        let div1_out1 = sub1_out1
            .div((constant182_out1).unsqueeze_dims(&[0isize, 1isize]));
        let transpose1_out1 = div1_out1.permute([2, 0, 1]);
        let shape1_out1: [i64; 3] = {
            let axes = &transpose1_out1.clone().dims()[0..3];
            let mut output = [0i64; 3];
            for i in 0..3 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let gather1_out1 = shape1_out1[1] as i64;
        let gather2_out1 = shape1_out1[2] as i64;
        let constant185_out1 = 1024i64;
        let sub2_out1 = constant185_out1 - gather1_out1;
        let constant186_out1 = 1024i64;
        let sub3_out1 = constant186_out1 - gather2_out1;
        let unsqueeze1_out1 = [sub3_out1 as i64];
        let unsqueeze2_out1 = [sub2_out1 as i64];
        let constant187_out1: [i64; 1] = [0i64];
        let constant189_out1: [i64; 1] = [0i64];
        let concat1_out1: [i64; 4usize] = [
            &constant187_out1[..],
            &unsqueeze1_out1[..],
            &constant189_out1[..],
            &unsqueeze2_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let unsqueeze3_out1 = [sub3_out1 as i64];
        let unsqueeze4_out1 = [sub2_out1 as i64];
        let constant191_out1: [i64; 1] = [0i64];
        let constant193_out1: [i64; 1] = [0i64];
        let concat2_out1: [i64; 4usize] = [
            &constant191_out1[..],
            &unsqueeze3_out1[..],
            &constant193_out1[..],
            &unsqueeze4_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let shape3_out1: [i64; 1] = [4i64];
        let constant195_out1: [i64; 1] = [0i64];
        let gather3_out1: [i64; 1usize] = constant195_out1
            .iter()
            .map(|&idx| {
                let actual_idx = if idx < 0 {
                    (shape3_out1.len() as i64 + idx) as usize
                } else {
                    idx as usize
                };
                shape3_out1[actual_idx]
            })
            .collect::<alloc::vec::Vec<_>>()
            .try_into()
            .unwrap();
        let constant196_out1 = 6i64;
        let sub4_out1 = {
            let mut result = gather3_out1;
            let scalar = constant196_out1;
            for result_item in result.iter_mut() {
                *result_item = scalar.saturating_sub(*result_item);
            }
            result
        };
        let constantofshape1_out1 = Tensor::<
            1,
            Int,
        >::from_data(
                burn::tensor::TensorData::from([0i64 as i64]),
                (&self.device, burn::tensor::DType::I64),
            )
            .reshape([1])
            .expand(sub4_out1);
        let concat3_out1 = {
            let shape_as_tensor_0: Tensor<1, Int> = Tensor::from_data(
                burn::tensor::TensorData::new(concat2_out1.to_vec(), [4]),
                (&self.device, burn::tensor::DType::I64),
            );
            burn::tensor::Tensor::cat(
                [shape_as_tensor_0, constantofshape1_out1].into(),
                0,
            )
        };
        let reshape1_out1 = concat3_out1.reshape([-1, 2]);
        let slice1_out1 = {
            let slice_input = reshape1_out1;
            let slice_dims = slice_input.dims();
            let reverse_bounds = |
                dim: usize,
                start: i64,
                end: i64,
            | -> core::ops::Range<usize> {
                if dim == 0 {
                    return 0..0;
                }
                let dim = dim as i64;
                let start = if start < 0 { start.saturating_add(dim) } else { start };
                let end = if end < 0 { end.saturating_add(dim) } else { end };
                let hi = start.clamp(-1, dim - 1) + 1;
                let lo = (end.clamp(-1, dim - 1) + 1).min(hi);
                (lo as usize)..(hi as usize)
            };
            slice_input
                .slice(
                    s![
                        reverse_bounds(slice_dims[0], - 1i64, - 9223372036854775807i64);
                        - 1, ..
                    ],
                )
        };
        let transpose2_out1 = slice1_out1.permute([1, 0]);
        let reshape2_out1 = transpose2_out1.reshape([-1]);
        let pad1_out1 = transpose1_out1
            .pad(
                {
                    let raw: alloc::vec::Vec<i64> = reshape2_out1
                        .to_data()
                        .convert::<i64>()
                        .try_into_vec::<i64>()
                        .unwrap();
                    assert_eq!(
                        raw.len(), 6usize,
                        "Pad: runtime pads length mismatch (expected {}, got {})",
                        6usize, raw.len(),
                    );
                    [
                        (
                            usize::try_from(raw[0usize])
                                .unwrap_or_else(|_| {
                                    panic!(
                                        "Pad: negative pad value {} at index {}", raw[0usize],
                                        0usize
                                    )
                                }),
                            usize::try_from(raw[3usize])
                                .unwrap_or_else(|_| {
                                    panic!(
                                        "Pad: negative pad value {} at index {}", raw[3usize],
                                        3usize
                                    )
                                }),
                        ),
                        (
                            usize::try_from(raw[1usize])
                                .unwrap_or_else(|_| {
                                    panic!(
                                        "Pad: negative pad value {} at index {}", raw[1usize],
                                        1usize
                                    )
                                }),
                            usize::try_from(raw[4usize])
                                .unwrap_or_else(|_| {
                                    panic!(
                                        "Pad: negative pad value {} at index {}", raw[4usize],
                                        4usize
                                    )
                                }),
                        ),
                        (
                            usize::try_from(raw[2usize])
                                .unwrap_or_else(|_| {
                                    panic!(
                                        "Pad: negative pad value {} at index {}", raw[2usize],
                                        2usize
                                    )
                                }),
                            usize::try_from(raw[5usize])
                                .unwrap_or_else(|_| {
                                    panic!(
                                        "Pad: negative pad value {} at index {}", raw[5usize],
                                        5usize
                                    )
                                }),
                        ),
                    ]
                },
                burn::tensor::ops::PadMode::Constant(0f32),
            );
        let unsqueeze5_out1: Tensor<4> = pad1_out1.unsqueeze_dims::<4>(&[0]);
        let conv2d1_out1 = self.conv2d1.forward(unsqueeze5_out1);
        let constant204_out1 = self.constant204.val();
        let div2_out1 = conv2d1_out1
            .clone()
            .div((constant204_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let erf1_out1 = div2_out1.erf();
        let constant205_out1 = self.constant205.val();
        let add1_out1 = erf1_out1
            .add((constant205_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let mul1_out1 = conv2d1_out1.mul(add1_out1);
        let constant206_out1 = self.constant206.val();
        let mul2_out1 = mul1_out1
            .mul((constant206_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let conv2d2_out1 = self.conv2d2.forward(mul2_out1);
        let conv2d3_out1 = self.conv2d3.forward(conv2d2_out1.clone());
        let constant207_out1 = self.constant207.val();
        let div3_out1 = conv2d3_out1
            .clone()
            .div((constant207_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let erf2_out1 = div3_out1.erf();
        let constant208_out1 = self.constant208.val();
        let add2_out1 = erf2_out1
            .add((constant208_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let mul3_out1 = conv2d3_out1.mul(add2_out1);
        let constant209_out1 = self.constant209.val();
        let mul4_out1 = mul3_out1
            .mul((constant209_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let conv2d4_out1 = self.conv2d4.forward(mul4_out1);
        let constant210_out1 = self.constant210.val();
        let div4_out1 = conv2d4_out1
            .clone()
            .div((constant210_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let erf3_out1 = div4_out1.erf();
        let constant211_out1 = self.constant211.val();
        let add3_out1 = erf3_out1
            .add((constant211_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let mul5_out1 = conv2d4_out1.mul(add3_out1);
        let constant212_out1 = self.constant212.val();
        let mul6_out1 = mul5_out1
            .mul((constant212_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let conv2d5_out1 = self.conv2d5.forward(mul6_out1);
        let add4_out1 = conv2d5_out1.add(conv2d2_out1);
        add4_out1
    }
}
#[derive(Module, Debug)]
pub struct Submodule2 {
    constant213: burn::module::Param<Tensor<1>>,
    constant214: burn::module::Param<Tensor<1>>,
    constant215: burn::module::Param<Tensor<1>>,
    conv2d6: Conv2d,
    constant216: burn::module::Param<Tensor<1>>,
    constant217: burn::module::Param<Tensor<1>>,
    constant218: burn::module::Param<Tensor<1>>,
    conv2d7: Conv2d,
    constant219: burn::module::Param<Tensor<1>>,
    constant220: burn::module::Param<Tensor<1>>,
    constant221: burn::module::Param<Tensor<1>>,
    conv2d8: Conv2d,
    constant222: burn::module::Param<Tensor<1>>,
    constant223: burn::module::Param<Tensor<1>>,
    constant224: burn::module::Param<Tensor<1>>,
    conv2d9: Conv2d,
    constant225: burn::module::Param<Tensor<1>>,
    constant226: burn::module::Param<Tensor<1>>,
    constant227: burn::module::Param<Tensor<1>>,
    conv2d10: Conv2d,
    constant228: burn::module::Param<Tensor<1>>,
    constant229: burn::module::Param<Tensor<1>>,
    constant230: burn::module::Param<Tensor<1>>,
    conv2d11: Conv2d,
    layernormalization1: LayerNorm,
    linear1: Linear,
    constant268: burn::module::Param<Tensor<1>>,
    constant269: burn::module::Param<Tensor<3>>,
    linear2: Linear,
    conv2d12: Conv2d,
    #[module(skip)]
    device: Device,
}
impl Submodule2 {
    #[allow(unused_variables)]
    pub fn new(device: &Device) -> Self {
        let constant213: burn::module::Param<Tensor<1>> = burn::module::Param::uninitialized(
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
        let constant214: burn::module::Param<Tensor<1>> = burn::module::Param::uninitialized(
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
        let constant215: burn::module::Param<Tensor<1>> = burn::module::Param::uninitialized(
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
        let conv2d6 = Conv2dConfig::new([64, 256], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let constant216: burn::module::Param<Tensor<1>> = burn::module::Param::uninitialized(
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
        let constant217: burn::module::Param<Tensor<1>> = burn::module::Param::uninitialized(
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
        let constant218: burn::module::Param<Tensor<1>> = burn::module::Param::uninitialized(
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
        let conv2d7 = Conv2dConfig::new([256, 256], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(256)
            .with_bias(true)
            .init(device);
        let constant219: burn::module::Param<Tensor<1>> = burn::module::Param::uninitialized(
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
        let constant220: burn::module::Param<Tensor<1>> = burn::module::Param::uninitialized(
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
        let constant221: burn::module::Param<Tensor<1>> = burn::module::Param::uninitialized(
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
        let conv2d8 = Conv2dConfig::new([256, 64], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let constant222: burn::module::Param<Tensor<1>> = burn::module::Param::uninitialized(
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
        let constant223: burn::module::Param<Tensor<1>> = burn::module::Param::uninitialized(
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
        let constant224: burn::module::Param<Tensor<1>> = burn::module::Param::uninitialized(
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
        let conv2d9 = Conv2dConfig::new([64, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let constant225: burn::module::Param<Tensor<1>> = burn::module::Param::uninitialized(
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
        let constant226: burn::module::Param<Tensor<1>> = burn::module::Param::uninitialized(
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
        let constant227: burn::module::Param<Tensor<1>> = burn::module::Param::uninitialized(
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
        let conv2d10 = Conv2dConfig::new([128, 128], [3, 3])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(128)
            .with_bias(true)
            .init(device);
        let constant228: burn::module::Param<Tensor<1>> = burn::module::Param::uninitialized(
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
        let constant229: burn::module::Param<Tensor<1>> = burn::module::Param::uninitialized(
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
        let constant230: burn::module::Param<Tensor<1>> = burn::module::Param::uninitialized(
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
        let conv2d11 = Conv2dConfig::new([128, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let layernormalization1 = LayerNormConfig::new(128)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(true)
            .init(device);
        let linear1 = LinearConfig::new(128, 384).with_bias(true).init(device);
        let constant268: burn::module::Param<Tensor<1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<
                1,
            >::from_data(
                burn::tensor::TensorData::from([0.1767766922712326f64]),
                (device, burn::tensor::DType::F32),
            ),
            device.clone(),
            false,
            [1].into(),
        );
        let constant269: burn::module::Param<Tensor<3>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<
                3,
            >::zeros([4, 49, 49], (device, burn::tensor::DType::F32)),
            device.clone(),
            false,
            [4, 49, 49].into(),
        );
        let linear2 = LinearConfig::new(128, 128).with_bias(true).init(device);
        let conv2d12 = Conv2dConfig::new([128, 128], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(128)
            .with_bias(true)
            .init(device);
        Self {
            constant213,
            constant214,
            constant215,
            conv2d6,
            constant216,
            constant217,
            constant218,
            conv2d7,
            constant219,
            constant220,
            constant221,
            conv2d8,
            constant222,
            constant223,
            constant224,
            conv2d9,
            constant225,
            constant226,
            constant227,
            conv2d10,
            constant228,
            constant229,
            constant230,
            conv2d11,
            layernormalization1,
            linear1,
            constant268,
            constant269,
            linear2,
            conv2d12,
            device: device.clone(),
        }
    }
    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(&self, add4_out1: Tensor<4>) -> Tensor<3> {
        let constant213_out1 = self.constant213.val();
        let div5_out1 = add4_out1
            .clone()
            .div((constant213_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let erf4_out1 = div5_out1.erf();
        let constant214_out1 = self.constant214.val();
        let add5_out1 = erf4_out1
            .add((constant214_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let mul7_out1 = add4_out1.mul(add5_out1);
        let constant215_out1 = self.constant215.val();
        let mul8_out1 = mul7_out1
            .mul((constant215_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let conv2d6_out1 = self.conv2d6.forward(mul8_out1.clone());
        let constant216_out1 = self.constant216.val();
        let div6_out1 = conv2d6_out1
            .clone()
            .div((constant216_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let erf5_out1 = div6_out1.erf();
        let constant217_out1 = self.constant217.val();
        let add6_out1 = erf5_out1
            .add((constant217_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let mul9_out1 = conv2d6_out1.mul(add6_out1);
        let constant218_out1 = self.constant218.val();
        let mul10_out1 = mul9_out1
            .mul((constant218_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let conv2d7_out1 = self.conv2d7.forward(mul10_out1);
        let constant219_out1 = self.constant219.val();
        let div7_out1 = conv2d7_out1
            .clone()
            .div((constant219_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let erf6_out1 = div7_out1.erf();
        let constant220_out1 = self.constant220.val();
        let add7_out1 = erf6_out1
            .add((constant220_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let mul11_out1 = conv2d7_out1.mul(add7_out1);
        let constant221_out1 = self.constant221.val();
        let mul12_out1 = mul11_out1
            .mul((constant221_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let conv2d8_out1 = self.conv2d8.forward(mul12_out1);
        let add8_out1 = conv2d8_out1.add(mul8_out1);
        let constant222_out1 = self.constant222.val();
        let div8_out1 = add8_out1
            .clone()
            .div((constant222_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let erf7_out1 = div8_out1.erf();
        let constant223_out1 = self.constant223.val();
        let add9_out1 = erf7_out1
            .add((constant223_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let mul13_out1 = add8_out1.mul(add9_out1);
        let constant224_out1 = self.constant224.val();
        let mul14_out1 = mul13_out1
            .mul((constant224_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let conv2d9_out1 = self.conv2d9.forward(mul14_out1);
        let constant225_out1 = self.constant225.val();
        let div9_out1 = conv2d9_out1
            .clone()
            .div((constant225_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let erf8_out1 = div9_out1.erf();
        let constant226_out1 = self.constant226.val();
        let add10_out1 = erf8_out1
            .add((constant226_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let mul15_out1 = conv2d9_out1.mul(add10_out1);
        let constant227_out1 = self.constant227.val();
        let mul16_out1 = mul15_out1
            .mul((constant227_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let conv2d10_out1 = self.conv2d10.forward(mul16_out1);
        let constant228_out1 = self.constant228.val();
        let div10_out1 = conv2d10_out1
            .clone()
            .div((constant228_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let erf9_out1 = div10_out1.erf();
        let constant229_out1 = self.constant229.val();
        let add11_out1 = erf9_out1
            .add((constant229_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let mul17_out1 = conv2d10_out1.mul(add11_out1);
        let constant230_out1 = self.constant230.val();
        let mul18_out1 = mul17_out1
            .mul((constant230_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let conv2d11_out1 = self.conv2d11.forward(mul18_out1);
        let shape4_out1: [i64; 4] = {
            let axes = &conv2d11_out1.clone().dims()[0..4];
            let mut output = [0i64; 4];
            for i in 0..4 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let slice2_out1: [i64; 2] = shape4_out1[0..2].try_into().unwrap();
        let constant234_out1: [i64; 1] = [-1i64];
        let concat4_out1: [i64; 3usize] = [&slice2_out1[..], &constant234_out1[..]]
            .concat()
            .try_into()
            .unwrap();
        let reshape3_out1 = conv2d11_out1.reshape(concat4_out1);
        let transpose3_out1 = reshape3_out1.permute([0, 2, 1]);
        let shape5_out1: [i64; 3] = {
            let axes = &transpose3_out1.clone().dims()[0..3];
            let mut output = [0i64; 3];
            for i in 0..3 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let gather4_out1 = shape5_out1[0] as i64;
        let gather5_out1 = shape5_out1[1] as i64;
        let gather6_out1 = shape5_out1[2] as i64;
        let unsqueeze6_out1 = [gather4_out1 as i64];
        let unsqueeze7_out1 = [gather6_out1 as i64];
        let constant239_out1: [i64; 1] = [128i64];
        let constant240_out1: [i64; 1] = [128i64];
        let concat5_out1: [i64; 4usize] = [
            &unsqueeze6_out1[..],
            &constant239_out1[..],
            &constant240_out1[..],
            &unsqueeze7_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape4_out1 = transpose3_out1.clone().reshape(concat5_out1);
        let constantofshape2_out1: [i64; 2usize] = [0i64, 0i64];
        let constant243_out1: [i64; 6] = [0i64, 0i64, 0i64, 5i64, 0i64, 5i64];
        let concat6_out1: [i64; 8usize] = [
            &constant243_out1[..],
            &constantofshape2_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape5_out1 = {
            let shape_array = concat6_out1 as [i64; 8usize];
            Tensor::<
                1,
                Int,
            >::from_data(
                TensorData::from(shape_array),
                (&self.device, burn::tensor::DType::I64),
            )
        }
            .reshape([-1, 2]);
        let slice3_out1 = {
            let slice_input = reshape5_out1;
            let slice_dims = slice_input.dims();
            let reverse_bounds = |
                dim: usize,
                start: i64,
                end: i64,
            | -> core::ops::Range<usize> {
                if dim == 0 {
                    return 0..0;
                }
                let dim = dim as i64;
                let start = if start < 0 { start.saturating_add(dim) } else { start };
                let end = if end < 0 { end.saturating_add(dim) } else { end };
                let hi = start.clamp(-1, dim - 1) + 1;
                let lo = (end.clamp(-1, dim - 1) + 1).min(hi);
                (lo as usize)..(hi as usize)
            };
            slice_input
                .slice(
                    s![
                        reverse_bounds(slice_dims[0], - 1i64, - 9223372036854775807i64);
                        - 1, ..
                    ],
                )
        };
        let transpose4_out1 = slice3_out1.permute([1, 0]);
        let reshape6_out1 = transpose4_out1.reshape([-1]);
        let pad2_out1 = reshape4_out1
            .pad(
                {
                    let raw: alloc::vec::Vec<i64> = reshape6_out1
                        .to_data()
                        .convert::<i64>()
                        .try_into_vec::<i64>()
                        .unwrap();
                    assert_eq!(
                        raw.len(), 8usize,
                        "Pad: runtime pads length mismatch (expected {}, got {})",
                        8usize, raw.len(),
                    );
                    [
                        (
                            usize::try_from(raw[0usize])
                                .unwrap_or_else(|_| {
                                    panic!(
                                        "Pad: negative pad value {} at index {}", raw[0usize],
                                        0usize
                                    )
                                }),
                            usize::try_from(raw[4usize])
                                .unwrap_or_else(|_| {
                                    panic!(
                                        "Pad: negative pad value {} at index {}", raw[4usize],
                                        4usize
                                    )
                                }),
                        ),
                        (
                            usize::try_from(raw[1usize])
                                .unwrap_or_else(|_| {
                                    panic!(
                                        "Pad: negative pad value {} at index {}", raw[1usize],
                                        1usize
                                    )
                                }),
                            usize::try_from(raw[5usize])
                                .unwrap_or_else(|_| {
                                    panic!(
                                        "Pad: negative pad value {} at index {}", raw[5usize],
                                        5usize
                                    )
                                }),
                        ),
                        (
                            usize::try_from(raw[2usize])
                                .unwrap_or_else(|_| {
                                    panic!(
                                        "Pad: negative pad value {} at index {}", raw[2usize],
                                        2usize
                                    )
                                }),
                            usize::try_from(raw[6usize])
                                .unwrap_or_else(|_| {
                                    panic!(
                                        "Pad: negative pad value {} at index {}", raw[6usize],
                                        6usize
                                    )
                                }),
                        ),
                        (
                            usize::try_from(raw[3usize])
                                .unwrap_or_else(|_| {
                                    panic!(
                                        "Pad: negative pad value {} at index {}", raw[3usize],
                                        3usize
                                    )
                                }),
                            usize::try_from(raw[7usize])
                                .unwrap_or_else(|_| {
                                    panic!(
                                        "Pad: negative pad value {} at index {}", raw[7usize],
                                        7usize
                                    )
                                }),
                        ),
                    ]
                },
                burn::tensor::ops::PadMode::Constant(0f32),
            );
        let unsqueeze8_out1 = [gather4_out1 as i64];
        let unsqueeze9_out1 = [gather6_out1 as i64];
        let constant251_out1: [i64; 1] = [19i64];
        let constant252_out1: [i64; 1] = [7i64];
        let constant253_out1: [i64; 1] = [19i64];
        let constant254_out1: [i64; 1] = [7i64];
        let concat7_out1: [i64; 6usize] = [
            &unsqueeze8_out1[..],
            &constant251_out1[..],
            &constant252_out1[..],
            &constant253_out1[..],
            &constant254_out1[..],
            &unsqueeze9_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape7_out1 = pad2_out1.reshape(concat7_out1);
        let transpose5_out1 = reshape7_out1.permute([0, 1, 3, 2, 4, 5]);
        let constant256_out1 = 19i64;
        let mul19_out1 = gather4_out1 * constant256_out1;
        let constant257_out1 = 19i64;
        let mul20_out1 = mul19_out1 * constant257_out1;
        let unsqueeze10_out1 = [mul20_out1 as i64];
        let unsqueeze11_out1 = [gather6_out1 as i64];
        let constant259_out1: [i64; 1] = [49i64];
        let concat8_out1: [i64; 3usize] = [
            &unsqueeze10_out1[..],
            &constant259_out1[..],
            &unsqueeze11_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape8_out1 = transpose5_out1.reshape(concat8_out1);
        let shape8_out1: [i64; 3] = {
            let axes = &reshape8_out1.clone().dims()[0..3];
            let mut output = [0i64; 3];
            for i in 0..3 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let gather7_out1 = shape8_out1[0] as i64;
        let gather8_out1 = shape8_out1[1] as i64;
        let layernormalization1_out1 = {
            let dtype = reshape8_out1.dtype();
            self.layernormalization1
                .forward(reshape8_out1.cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let linear1_out1 = self.linear1.forward(layernormalization1_out1);
        let unsqueeze12_out1 = [gather7_out1 as i64];
        let unsqueeze13_out1 = [gather8_out1 as i64];
        let constant266_out1: [i64; 1] = [-1i64];
        let constant265_out1: [i64; 1] = [4i64];
        let concat9_out1: [i64; 4usize] = [
            &unsqueeze12_out1[..],
            &unsqueeze13_out1[..],
            &constant265_out1[..],
            &constant266_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape9_out1 = linear1_out1.reshape(concat9_out1);
        let split_tensors = reshape9_out1.split_with_sizes([32, 32, 32].into(), 3);
        let [split1_out1, split1_out2, split1_out3] = split_tensors.try_into().unwrap();
        let transpose6_out1 = split1_out1.permute([0, 2, 1, 3]);
        let transpose7_out1 = split1_out3.permute([0, 2, 1, 3]);
        let transpose8_out1 = split1_out2.permute([0, 2, 3, 1]);
        let matmul2_out1 = transpose6_out1.matmul(transpose8_out1);
        let constant268_out1 = self.constant268.val();
        let mul21_out1 = matmul2_out1
            .mul((constant268_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let constant269_out1 = self.constant269.val();
        let add12_out1 = mul21_out1.add((constant269_out1).unsqueeze_dims(&[0isize]));
        let softmax1_out1 = burn::tensor::activation::softmax(add12_out1, 3);
        let matmul3_out1 = softmax1_out1.matmul(transpose7_out1);
        let transpose9_out1 = matmul3_out1.permute([0, 2, 1, 3]);
        let unsqueeze14_out1 = [gather7_out1 as i64];
        let unsqueeze15_out1 = [gather8_out1 as i64];
        let constant272_out1: [i64; 1] = [128i64];
        let concat10_out1: [i64; 3usize] = [
            &unsqueeze14_out1[..],
            &unsqueeze15_out1[..],
            &constant272_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape10_out1 = transpose9_out1.reshape(concat10_out1);
        let linear2_out1 = self.linear2.forward(reshape10_out1);
        let unsqueeze16_out1 = [gather4_out1 as i64];
        let unsqueeze17_out1 = [gather6_out1 as i64];
        let constant274_out1: [i64; 1] = [19i64];
        let constant275_out1: [i64; 1] = [19i64];
        let constant276_out1: [i64; 1] = [7i64];
        let constant277_out1: [i64; 1] = [7i64];
        let concat11_out1: [i64; 6usize] = [
            &unsqueeze16_out1[..],
            &constant274_out1[..],
            &constant275_out1[..],
            &constant276_out1[..],
            &constant277_out1[..],
            &unsqueeze17_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape11_out1 = linear2_out1.reshape(concat11_out1);
        let transpose10_out1 = reshape11_out1.permute([0, 1, 3, 2, 4, 5]);
        let unsqueeze18_out1 = [gather4_out1 as i64];
        let unsqueeze19_out1 = [gather6_out1 as i64];
        let constant280_out1: [i64; 1] = [133i64];
        let constant281_out1: [i64; 1] = [133i64];
        let concat12_out1: [i64; 4usize] = [
            &unsqueeze18_out1[..],
            &constant280_out1[..],
            &constant281_out1[..],
            &unsqueeze19_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape12_out1 = transpose10_out1.reshape(concat12_out1);
        let slice4_out1 = reshape12_out1.slice(s![.., 0..128, .., ..]);
        let slice5_out1 = slice4_out1.slice(s![.., .., 0..128, ..]);
        let unsqueeze20_out1 = [gather4_out1 as i64];
        let unsqueeze21_out1 = [gather5_out1 as i64];
        let unsqueeze22_out1 = [gather6_out1 as i64];
        let concat13_out1: [i64; 3usize] = [
            &unsqueeze20_out1[..],
            &unsqueeze21_out1[..],
            &unsqueeze22_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape13_out1 = slice5_out1.reshape(concat13_out1);
        let add13_out1 = transpose3_out1.add(reshape13_out1);
        let transpose11_out1 = add13_out1.permute([0, 2, 1]);
        let unsqueeze23_out1 = [gather4_out1 as i64];
        let unsqueeze24_out1 = [gather6_out1 as i64];
        let constant297_out1: [i64; 1] = [128i64];
        let constant296_out1: [i64; 1] = [128i64];
        let concat14_out1: [i64; 4usize] = [
            &unsqueeze23_out1[..],
            &unsqueeze24_out1[..],
            &constant296_out1[..],
            &constant297_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape14_out1 = transpose11_out1.reshape(concat14_out1);
        let conv2d12_out1 = self.conv2d12.forward(reshape14_out1);
        let unsqueeze25_out1 = [gather4_out1 as i64];
        let unsqueeze26_out1 = [gather6_out1 as i64];
        let unsqueeze27_out1 = [gather5_out1 as i64];
        let concat15_out1: [i64; 3usize] = [
            &unsqueeze25_out1[..],
            &unsqueeze26_out1[..],
            &unsqueeze27_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape15_out1 = conv2d12_out1.reshape(concat15_out1);
        reshape15_out1
    }
}
#[derive(Module, Debug)]
pub struct Submodule3 {
    layernormalization2: LayerNorm,
    linear3: Linear,
    constant301: burn::module::Param<Tensor<1>>,
    constant302: burn::module::Param<Tensor<1>>,
    constant303: burn::module::Param<Tensor<1>>,
    linear4: Linear,
    layernormalization3: LayerNorm,
    linear5: Linear,
    constant336: burn::module::Param<Tensor<1>>,
    constant337: burn::module::Param<Tensor<3>>,
    linear6: Linear,
    conv2d13: Conv2d,
    #[module(skip)]
    device: Device,
}
impl Submodule3 {
    #[allow(unused_variables)]
    pub fn new(device: &Device) -> Self {
        let layernormalization2 = LayerNormConfig::new(128)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(true)
            .init(device);
        let linear3 = LinearConfig::new(128, 512).with_bias(true).init(device);
        let constant301: burn::module::Param<Tensor<1>> = burn::module::Param::uninitialized(
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
        let constant302: burn::module::Param<Tensor<1>> = burn::module::Param::uninitialized(
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
        let constant303: burn::module::Param<Tensor<1>> = burn::module::Param::uninitialized(
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
        let linear4 = LinearConfig::new(512, 128).with_bias(true).init(device);
        let layernormalization3 = LayerNormConfig::new(128)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(true)
            .init(device);
        let linear5 = LinearConfig::new(128, 384).with_bias(true).init(device);
        let constant336: burn::module::Param<Tensor<1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<
                1,
            >::from_data(
                burn::tensor::TensorData::from([0.1767766922712326f64]),
                (device, burn::tensor::DType::F32),
            ),
            device.clone(),
            false,
            [1].into(),
        );
        let constant337: burn::module::Param<Tensor<3>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<
                3,
            >::zeros([4, 49, 49], (device, burn::tensor::DType::F32)),
            device.clone(),
            false,
            [4, 49, 49].into(),
        );
        let linear6 = LinearConfig::new(128, 128).with_bias(true).init(device);
        let conv2d13 = Conv2dConfig::new([128, 128], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(128)
            .with_bias(true)
            .init(device);
        Self {
            layernormalization2,
            linear3,
            constant301,
            constant302,
            constant303,
            linear4,
            layernormalization3,
            linear5,
            constant336,
            constant337,
            linear6,
            conv2d13,
            device: device.clone(),
        }
    }
    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(&self, reshape15_out1: Tensor<3>) -> Tensor<3> {
        let transpose12_out1 = reshape15_out1.permute([0, 2, 1]);
        let layernormalization2_out1 = {
            let dtype = transpose12_out1.clone().dtype();
            self.layernormalization2
                .forward(transpose12_out1.clone().cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let linear3_out1 = self.linear3.forward(layernormalization2_out1);
        let constant301_out1 = self.constant301.val();
        let div11_out1 = linear3_out1
            .clone()
            .div((constant301_out1).unsqueeze_dims(&[0isize, 1isize]));
        let erf10_out1 = div11_out1.erf();
        let constant302_out1 = self.constant302.val();
        let add14_out1 = erf10_out1
            .add((constant302_out1).unsqueeze_dims(&[0isize, 1isize]));
        let mul22_out1 = linear3_out1.mul(add14_out1);
        let constant303_out1 = self.constant303.val();
        let mul23_out1 = mul22_out1
            .mul((constant303_out1).unsqueeze_dims(&[0isize, 1isize]));
        let linear4_out1 = self.linear4.forward(mul23_out1);
        let add15_out1 = transpose12_out1.add(linear4_out1);
        let shape10_out1: [i64; 3] = {
            let axes = &add15_out1.clone().dims()[0..3];
            let mut output = [0i64; 3];
            for i in 0..3 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let gather9_out1 = shape10_out1[0] as i64;
        let gather10_out1 = shape10_out1[1] as i64;
        let gather11_out1 = shape10_out1[2] as i64;
        let unsqueeze28_out1 = [gather9_out1 as i64];
        let unsqueeze29_out1 = [gather11_out1 as i64];
        let constant308_out1: [i64; 1] = [128i64];
        let constant309_out1: [i64; 1] = [128i64];
        let concat16_out1: [i64; 4usize] = [
            &unsqueeze28_out1[..],
            &constant308_out1[..],
            &constant309_out1[..],
            &unsqueeze29_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape16_out1 = add15_out1.clone().reshape(concat16_out1);
        let constantofshape3_out1: [i64; 2usize] = [0i64, 0i64];
        let constant312_out1: [i64; 6] = [0i64, 0i64, 0i64, 5i64, 0i64, 5i64];
        let concat17_out1: [i64; 8usize] = [
            &constant312_out1[..],
            &constantofshape3_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape17_out1 = {
            let shape_array = concat17_out1 as [i64; 8usize];
            Tensor::<
                1,
                Int,
            >::from_data(
                TensorData::from(shape_array),
                (&self.device, burn::tensor::DType::I64),
            )
        }
            .reshape([-1, 2]);
        let slice6_out1 = {
            let slice_input = reshape17_out1;
            let slice_dims = slice_input.dims();
            let reverse_bounds = |
                dim: usize,
                start: i64,
                end: i64,
            | -> core::ops::Range<usize> {
                if dim == 0 {
                    return 0..0;
                }
                let dim = dim as i64;
                let start = if start < 0 { start.saturating_add(dim) } else { start };
                let end = if end < 0 { end.saturating_add(dim) } else { end };
                let hi = start.clamp(-1, dim - 1) + 1;
                let lo = (end.clamp(-1, dim - 1) + 1).min(hi);
                (lo as usize)..(hi as usize)
            };
            slice_input
                .slice(
                    s![
                        reverse_bounds(slice_dims[0], - 1i64, - 9223372036854775807i64);
                        - 1, ..
                    ],
                )
        };
        let transpose13_out1 = slice6_out1.permute([1, 0]);
        let reshape18_out1 = transpose13_out1.reshape([-1]);
        let pad3_out1 = reshape16_out1
            .pad(
                {
                    let raw: alloc::vec::Vec<i64> = reshape18_out1
                        .to_data()
                        .convert::<i64>()
                        .try_into_vec::<i64>()
                        .unwrap();
                    assert_eq!(
                        raw.len(), 8usize,
                        "Pad: runtime pads length mismatch (expected {}, got {})",
                        8usize, raw.len(),
                    );
                    [
                        (
                            usize::try_from(raw[0usize])
                                .unwrap_or_else(|_| {
                                    panic!(
                                        "Pad: negative pad value {} at index {}", raw[0usize],
                                        0usize
                                    )
                                }),
                            usize::try_from(raw[4usize])
                                .unwrap_or_else(|_| {
                                    panic!(
                                        "Pad: negative pad value {} at index {}", raw[4usize],
                                        4usize
                                    )
                                }),
                        ),
                        (
                            usize::try_from(raw[1usize])
                                .unwrap_or_else(|_| {
                                    panic!(
                                        "Pad: negative pad value {} at index {}", raw[1usize],
                                        1usize
                                    )
                                }),
                            usize::try_from(raw[5usize])
                                .unwrap_or_else(|_| {
                                    panic!(
                                        "Pad: negative pad value {} at index {}", raw[5usize],
                                        5usize
                                    )
                                }),
                        ),
                        (
                            usize::try_from(raw[2usize])
                                .unwrap_or_else(|_| {
                                    panic!(
                                        "Pad: negative pad value {} at index {}", raw[2usize],
                                        2usize
                                    )
                                }),
                            usize::try_from(raw[6usize])
                                .unwrap_or_else(|_| {
                                    panic!(
                                        "Pad: negative pad value {} at index {}", raw[6usize],
                                        6usize
                                    )
                                }),
                        ),
                        (
                            usize::try_from(raw[3usize])
                                .unwrap_or_else(|_| {
                                    panic!(
                                        "Pad: negative pad value {} at index {}", raw[3usize],
                                        3usize
                                    )
                                }),
                            usize::try_from(raw[7usize])
                                .unwrap_or_else(|_| {
                                    panic!(
                                        "Pad: negative pad value {} at index {}", raw[7usize],
                                        7usize
                                    )
                                }),
                        ),
                    ]
                },
                burn::tensor::ops::PadMode::Constant(0f32),
            );
        let unsqueeze30_out1 = [gather9_out1 as i64];
        let unsqueeze31_out1 = [gather11_out1 as i64];
        let constant320_out1: [i64; 1] = [19i64];
        let constant321_out1: [i64; 1] = [7i64];
        let constant322_out1: [i64; 1] = [19i64];
        let constant323_out1: [i64; 1] = [7i64];
        let concat18_out1: [i64; 6usize] = [
            &unsqueeze30_out1[..],
            &constant320_out1[..],
            &constant321_out1[..],
            &constant322_out1[..],
            &constant323_out1[..],
            &unsqueeze31_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape19_out1 = pad3_out1.reshape(concat18_out1);
        let transpose14_out1 = reshape19_out1.permute([0, 1, 3, 2, 4, 5]);
        let constant325_out1 = 19i64;
        let mul24_out1 = gather9_out1 * constant325_out1;
        let constant326_out1 = 19i64;
        let mul25_out1 = mul24_out1 * constant326_out1;
        let unsqueeze32_out1 = [mul25_out1 as i64];
        let unsqueeze33_out1 = [gather11_out1 as i64];
        let constant328_out1: [i64; 1] = [49i64];
        let concat19_out1: [i64; 3usize] = [
            &unsqueeze32_out1[..],
            &constant328_out1[..],
            &unsqueeze33_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape20_out1 = transpose14_out1.reshape(concat19_out1);
        let shape13_out1: [i64; 3] = {
            let axes = &reshape20_out1.clone().dims()[0..3];
            let mut output = [0i64; 3];
            for i in 0..3 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let gather12_out1 = shape13_out1[0] as i64;
        let gather13_out1 = shape13_out1[1] as i64;
        let layernormalization3_out1 = {
            let dtype = reshape20_out1.dtype();
            self.layernormalization3
                .forward(reshape20_out1.cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let linear5_out1 = self.linear5.forward(layernormalization3_out1);
        let unsqueeze34_out1 = [gather12_out1 as i64];
        let unsqueeze35_out1 = [gather13_out1 as i64];
        let constant335_out1: [i64; 1] = [-1i64];
        let constant334_out1: [i64; 1] = [4i64];
        let concat20_out1: [i64; 4usize] = [
            &unsqueeze34_out1[..],
            &unsqueeze35_out1[..],
            &constant334_out1[..],
            &constant335_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape21_out1 = linear5_out1.reshape(concat20_out1);
        let split_tensors = reshape21_out1.split_with_sizes([32, 32, 32].into(), 3);
        let [split2_out1, split2_out2, split2_out3] = split_tensors.try_into().unwrap();
        let transpose15_out1 = split2_out1.permute([0, 2, 1, 3]);
        let transpose16_out1 = split2_out3.permute([0, 2, 1, 3]);
        let transpose17_out1 = split2_out2.permute([0, 2, 3, 1]);
        let matmul8_out1 = transpose15_out1.matmul(transpose17_out1);
        let constant336_out1 = self.constant336.val();
        let mul26_out1 = matmul8_out1
            .mul((constant336_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let constant337_out1 = self.constant337.val();
        let add16_out1 = mul26_out1.add((constant337_out1).unsqueeze_dims(&[0isize]));
        let softmax2_out1 = burn::tensor::activation::softmax(add16_out1, 3);
        let matmul9_out1 = softmax2_out1.matmul(transpose16_out1);
        let transpose18_out1 = matmul9_out1.permute([0, 2, 1, 3]);
        let unsqueeze36_out1 = [gather12_out1 as i64];
        let unsqueeze37_out1 = [gather13_out1 as i64];
        let constant340_out1: [i64; 1] = [128i64];
        let concat21_out1: [i64; 3usize] = [
            &unsqueeze36_out1[..],
            &unsqueeze37_out1[..],
            &constant340_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape22_out1 = transpose18_out1.reshape(concat21_out1);
        let linear6_out1 = self.linear6.forward(reshape22_out1);
        let unsqueeze38_out1 = [gather9_out1 as i64];
        let unsqueeze39_out1 = [gather11_out1 as i64];
        let constant342_out1: [i64; 1] = [19i64];
        let constant343_out1: [i64; 1] = [19i64];
        let constant344_out1: [i64; 1] = [7i64];
        let constant345_out1: [i64; 1] = [7i64];
        let concat22_out1: [i64; 6usize] = [
            &unsqueeze38_out1[..],
            &constant342_out1[..],
            &constant343_out1[..],
            &constant344_out1[..],
            &constant345_out1[..],
            &unsqueeze39_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape23_out1 = linear6_out1.reshape(concat22_out1);
        let transpose19_out1 = reshape23_out1.permute([0, 1, 3, 2, 4, 5]);
        let unsqueeze40_out1 = [gather9_out1 as i64];
        let unsqueeze41_out1 = [gather11_out1 as i64];
        let constant348_out1: [i64; 1] = [133i64];
        let constant349_out1: [i64; 1] = [133i64];
        let concat23_out1: [i64; 4usize] = [
            &unsqueeze40_out1[..],
            &constant348_out1[..],
            &constant349_out1[..],
            &unsqueeze41_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape24_out1 = transpose19_out1.reshape(concat23_out1);
        let slice7_out1 = reshape24_out1.slice(s![.., 0..128, .., ..]);
        let slice8_out1 = slice7_out1.slice(s![.., .., 0..128, ..]);
        let unsqueeze42_out1 = [gather9_out1 as i64];
        let unsqueeze43_out1 = [gather10_out1 as i64];
        let unsqueeze44_out1 = [gather11_out1 as i64];
        let concat24_out1: [i64; 3usize] = [
            &unsqueeze42_out1[..],
            &unsqueeze43_out1[..],
            &unsqueeze44_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape25_out1 = slice8_out1.reshape(concat24_out1);
        let add17_out1 = add15_out1.add(reshape25_out1);
        let transpose20_out1 = add17_out1.permute([0, 2, 1]);
        let unsqueeze45_out1 = [gather9_out1 as i64];
        let unsqueeze46_out1 = [gather11_out1 as i64];
        let constant365_out1: [i64; 1] = [128i64];
        let constant364_out1: [i64; 1] = [128i64];
        let concat25_out1: [i64; 4usize] = [
            &unsqueeze45_out1[..],
            &unsqueeze46_out1[..],
            &constant364_out1[..],
            &constant365_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape26_out1 = transpose20_out1.reshape(concat25_out1);
        let conv2d13_out1 = self.conv2d13.forward(reshape26_out1);
        let unsqueeze47_out1 = [gather9_out1 as i64];
        let unsqueeze48_out1 = [gather11_out1 as i64];
        let unsqueeze49_out1 = [gather10_out1 as i64];
        let concat26_out1: [i64; 3usize] = [
            &unsqueeze47_out1[..],
            &unsqueeze48_out1[..],
            &unsqueeze49_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape27_out1 = conv2d13_out1.reshape(concat26_out1);
        reshape27_out1
    }
}
#[derive(Module, Debug)]
pub struct Submodule4 {
    layernormalization4: LayerNorm,
    linear7: Linear,
    constant369: burn::module::Param<Tensor<1>>,
    constant370: burn::module::Param<Tensor<1>>,
    constant371: burn::module::Param<Tensor<1>>,
    linear8: Linear,
    conv2d14: Conv2d,
    constant373: burn::module::Param<Tensor<1>>,
    constant374: burn::module::Param<Tensor<1>>,
    constant375: burn::module::Param<Tensor<1>>,
    conv2d15: Conv2d,
    constant376: burn::module::Param<Tensor<1>>,
    constant377: burn::module::Param<Tensor<1>>,
    constant378: burn::module::Param<Tensor<1>>,
    conv2d16: Conv2d,
    layernormalization5: LayerNorm,
    linear9: Linear,
    constant415: burn::module::Param<Tensor<1>>,
    constant416: burn::module::Param<Tensor<3>>,
    linear10: Linear,
    conv2d17: Conv2d,
    #[module(skip)]
    device: Device,
}
impl Submodule4 {
    #[allow(unused_variables)]
    pub fn new(device: &Device) -> Self {
        let layernormalization4 = LayerNormConfig::new(128)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(true)
            .init(device);
        let linear7 = LinearConfig::new(128, 512).with_bias(true).init(device);
        let constant369: burn::module::Param<Tensor<1>> = burn::module::Param::uninitialized(
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
        let constant370: burn::module::Param<Tensor<1>> = burn::module::Param::uninitialized(
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
        let constant371: burn::module::Param<Tensor<1>> = burn::module::Param::uninitialized(
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
        let linear8 = LinearConfig::new(512, 128).with_bias(true).init(device);
        let conv2d14 = Conv2dConfig::new([128, 160], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let constant373: burn::module::Param<Tensor<1>> = burn::module::Param::uninitialized(
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
        let constant374: burn::module::Param<Tensor<1>> = burn::module::Param::uninitialized(
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
        let constant375: burn::module::Param<Tensor<1>> = burn::module::Param::uninitialized(
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
        let conv2d15 = Conv2dConfig::new([160, 160], [3, 3])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(160)
            .with_bias(true)
            .init(device);
        let constant376: burn::module::Param<Tensor<1>> = burn::module::Param::uninitialized(
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
        let constant377: burn::module::Param<Tensor<1>> = burn::module::Param::uninitialized(
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
        let constant378: burn::module::Param<Tensor<1>> = burn::module::Param::uninitialized(
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
        let conv2d16 = Conv2dConfig::new([160, 160], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let layernormalization5 = LayerNormConfig::new(160)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(true)
            .init(device);
        let linear9 = LinearConfig::new(160, 480).with_bias(true).init(device);
        let constant415: burn::module::Param<Tensor<1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<
                1,
            >::from_data(
                burn::tensor::TensorData::from([0.1767766922712326f64]),
                (device, burn::tensor::DType::F32),
            ),
            device.clone(),
            false,
            [1].into(),
        );
        let constant416: burn::module::Param<Tensor<3>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<
                3,
            >::zeros([5, 196, 196], (device, burn::tensor::DType::F32)),
            device.clone(),
            false,
            [5, 196, 196].into(),
        );
        let linear10 = LinearConfig::new(160, 160).with_bias(true).init(device);
        let conv2d17 = Conv2dConfig::new([160, 160], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(160)
            .with_bias(true)
            .init(device);
        Self {
            layernormalization4,
            linear7,
            constant369,
            constant370,
            constant371,
            linear8,
            conv2d14,
            constant373,
            constant374,
            constant375,
            conv2d15,
            constant376,
            constant377,
            constant378,
            conv2d16,
            layernormalization5,
            linear9,
            constant415,
            constant416,
            linear10,
            conv2d17,
            device: device.clone(),
        }
    }
    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(&self, reshape27_out1: Tensor<3>) -> Tensor<3> {
        let transpose21_out1 = reshape27_out1.permute([0, 2, 1]);
        let layernormalization4_out1 = {
            let dtype = transpose21_out1.clone().dtype();
            self.layernormalization4
                .forward(transpose21_out1.clone().cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let linear7_out1 = self.linear7.forward(layernormalization4_out1);
        let constant369_out1 = self.constant369.val();
        let div12_out1 = linear7_out1
            .clone()
            .div((constant369_out1).unsqueeze_dims(&[0isize, 1isize]));
        let erf11_out1 = div12_out1.erf();
        let constant370_out1 = self.constant370.val();
        let add18_out1 = erf11_out1
            .add((constant370_out1).unsqueeze_dims(&[0isize, 1isize]));
        let mul27_out1 = linear7_out1.mul(add18_out1);
        let constant371_out1 = self.constant371.val();
        let mul28_out1 = mul27_out1
            .mul((constant371_out1).unsqueeze_dims(&[0isize, 1isize]));
        let linear8_out1 = self.linear8.forward(mul28_out1);
        let add19_out1 = transpose21_out1.add(linear8_out1);
        let reshape28_out1 = add19_out1.reshape([1, 128, 128, -1]);
        let transpose22_out1 = reshape28_out1.permute([0, 3, 1, 2]);
        let conv2d14_out1 = self.conv2d14.forward(transpose22_out1);
        let constant373_out1 = self.constant373.val();
        let div13_out1 = conv2d14_out1
            .clone()
            .div((constant373_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let erf12_out1 = div13_out1.erf();
        let constant374_out1 = self.constant374.val();
        let add20_out1 = erf12_out1
            .add((constant374_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let mul29_out1 = conv2d14_out1.mul(add20_out1);
        let constant375_out1 = self.constant375.val();
        let mul30_out1 = mul29_out1
            .mul((constant375_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let conv2d15_out1 = self.conv2d15.forward(mul30_out1);
        let constant376_out1 = self.constant376.val();
        let div14_out1 = conv2d15_out1
            .clone()
            .div((constant376_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let erf13_out1 = div14_out1.erf();
        let constant377_out1 = self.constant377.val();
        let add21_out1 = erf13_out1
            .add((constant377_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let mul31_out1 = conv2d15_out1.mul(add21_out1);
        let constant378_out1 = self.constant378.val();
        let mul32_out1 = mul31_out1
            .mul((constant378_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let conv2d16_out1 = self.conv2d16.forward(mul32_out1);
        let reshape29_out1 = conv2d16_out1.reshape([1, 160, -1]);
        let transpose23_out1 = reshape29_out1.permute([0, 2, 1]);
        let shape16_out1: [i64; 3] = {
            let axes = &transpose23_out1.clone().dims()[0..3];
            let mut output = [0i64; 3];
            for i in 0..3 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let gather14_out1 = shape16_out1[0] as i64;
        let gather15_out1 = shape16_out1[1] as i64;
        let gather16_out1 = shape16_out1[2] as i64;
        let unsqueeze50_out1 = [gather14_out1 as i64];
        let unsqueeze51_out1 = [gather16_out1 as i64];
        let constant387_out1: [i64; 1] = [64i64];
        let constant388_out1: [i64; 1] = [64i64];
        let concat28_out1: [i64; 4usize] = [
            &unsqueeze50_out1[..],
            &constant387_out1[..],
            &constant388_out1[..],
            &unsqueeze51_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape30_out1 = transpose23_out1.clone().reshape(concat28_out1);
        let constantofshape4_out1: [i64; 2usize] = [0i64, 0i64];
        let constant391_out1: [i64; 6] = [0i64, 0i64, 0i64, 6i64, 0i64, 6i64];
        let concat29_out1: [i64; 8usize] = [
            &constant391_out1[..],
            &constantofshape4_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape31_out1 = {
            let shape_array = concat29_out1 as [i64; 8usize];
            Tensor::<
                1,
                Int,
            >::from_data(
                TensorData::from(shape_array),
                (&self.device, burn::tensor::DType::I64),
            )
        }
            .reshape([-1, 2]);
        let slice10_out1 = {
            let slice_input = reshape31_out1;
            let slice_dims = slice_input.dims();
            let reverse_bounds = |
                dim: usize,
                start: i64,
                end: i64,
            | -> core::ops::Range<usize> {
                if dim == 0 {
                    return 0..0;
                }
                let dim = dim as i64;
                let start = if start < 0 { start.saturating_add(dim) } else { start };
                let end = if end < 0 { end.saturating_add(dim) } else { end };
                let hi = start.clamp(-1, dim - 1) + 1;
                let lo = (end.clamp(-1, dim - 1) + 1).min(hi);
                (lo as usize)..(hi as usize)
            };
            slice_input
                .slice(
                    s![
                        reverse_bounds(slice_dims[0], - 1i64, - 9223372036854775807i64);
                        - 1, ..
                    ],
                )
        };
        let transpose24_out1 = slice10_out1.permute([1, 0]);
        let reshape32_out1 = transpose24_out1.reshape([-1]);
        let pad4_out1 = reshape30_out1
            .pad(
                {
                    let raw: alloc::vec::Vec<i64> = reshape32_out1
                        .to_data()
                        .convert::<i64>()
                        .try_into_vec::<i64>()
                        .unwrap();
                    assert_eq!(
                        raw.len(), 8usize,
                        "Pad: runtime pads length mismatch (expected {}, got {})",
                        8usize, raw.len(),
                    );
                    [
                        (
                            usize::try_from(raw[0usize])
                                .unwrap_or_else(|_| {
                                    panic!(
                                        "Pad: negative pad value {} at index {}", raw[0usize],
                                        0usize
                                    )
                                }),
                            usize::try_from(raw[4usize])
                                .unwrap_or_else(|_| {
                                    panic!(
                                        "Pad: negative pad value {} at index {}", raw[4usize],
                                        4usize
                                    )
                                }),
                        ),
                        (
                            usize::try_from(raw[1usize])
                                .unwrap_or_else(|_| {
                                    panic!(
                                        "Pad: negative pad value {} at index {}", raw[1usize],
                                        1usize
                                    )
                                }),
                            usize::try_from(raw[5usize])
                                .unwrap_or_else(|_| {
                                    panic!(
                                        "Pad: negative pad value {} at index {}", raw[5usize],
                                        5usize
                                    )
                                }),
                        ),
                        (
                            usize::try_from(raw[2usize])
                                .unwrap_or_else(|_| {
                                    panic!(
                                        "Pad: negative pad value {} at index {}", raw[2usize],
                                        2usize
                                    )
                                }),
                            usize::try_from(raw[6usize])
                                .unwrap_or_else(|_| {
                                    panic!(
                                        "Pad: negative pad value {} at index {}", raw[6usize],
                                        6usize
                                    )
                                }),
                        ),
                        (
                            usize::try_from(raw[3usize])
                                .unwrap_or_else(|_| {
                                    panic!(
                                        "Pad: negative pad value {} at index {}", raw[3usize],
                                        3usize
                                    )
                                }),
                            usize::try_from(raw[7usize])
                                .unwrap_or_else(|_| {
                                    panic!(
                                        "Pad: negative pad value {} at index {}", raw[7usize],
                                        7usize
                                    )
                                }),
                        ),
                    ]
                },
                burn::tensor::ops::PadMode::Constant(0f32),
            );
        let unsqueeze52_out1 = [gather14_out1 as i64];
        let unsqueeze53_out1 = [gather16_out1 as i64];
        let constant399_out1: [i64; 1] = [5i64];
        let constant400_out1: [i64; 1] = [14i64];
        let constant401_out1: [i64; 1] = [5i64];
        let constant402_out1: [i64; 1] = [14i64];
        let concat30_out1: [i64; 6usize] = [
            &unsqueeze52_out1[..],
            &constant399_out1[..],
            &constant400_out1[..],
            &constant401_out1[..],
            &constant402_out1[..],
            &unsqueeze53_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape33_out1 = pad4_out1.reshape(concat30_out1);
        let transpose25_out1 = reshape33_out1.permute([0, 1, 3, 2, 4, 5]);
        let constant404_out1 = 5i64;
        let mul33_out1 = gather14_out1 * constant404_out1;
        let constant405_out1 = 5i64;
        let mul34_out1 = mul33_out1 * constant405_out1;
        let unsqueeze54_out1 = [mul34_out1 as i64];
        let unsqueeze55_out1 = [gather16_out1 as i64];
        let constant407_out1: [i64; 1] = [196i64];
        let concat31_out1: [i64; 3usize] = [
            &unsqueeze54_out1[..],
            &constant407_out1[..],
            &unsqueeze55_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape34_out1 = transpose25_out1.reshape(concat31_out1);
        let shape19_out1: [i64; 3] = {
            let axes = &reshape34_out1.clone().dims()[0..3];
            let mut output = [0i64; 3];
            for i in 0..3 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let gather17_out1 = shape19_out1[0] as i64;
        let gather18_out1 = shape19_out1[1] as i64;
        let layernormalization5_out1 = {
            let dtype = reshape34_out1.dtype();
            self.layernormalization5
                .forward(reshape34_out1.cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let linear9_out1 = self.linear9.forward(layernormalization5_out1);
        let unsqueeze56_out1 = [gather17_out1 as i64];
        let unsqueeze57_out1 = [gather18_out1 as i64];
        let constant414_out1: [i64; 1] = [-1i64];
        let constant413_out1: [i64; 1] = [5i64];
        let concat32_out1: [i64; 4usize] = [
            &unsqueeze56_out1[..],
            &unsqueeze57_out1[..],
            &constant413_out1[..],
            &constant414_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape35_out1 = linear9_out1.reshape(concat32_out1);
        let split_tensors = reshape35_out1.split_with_sizes([32, 32, 32].into(), 3);
        let [split3_out1, split3_out2, split3_out3] = split_tensors.try_into().unwrap();
        let transpose26_out1 = split3_out1.permute([0, 2, 1, 3]);
        let transpose27_out1 = split3_out3.permute([0, 2, 1, 3]);
        let transpose28_out1 = split3_out2.permute([0, 2, 3, 1]);
        let matmul14_out1 = transpose26_out1.matmul(transpose28_out1);
        let constant415_out1 = self.constant415.val();
        let mul35_out1 = matmul14_out1
            .mul((constant415_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let constant416_out1 = self.constant416.val();
        let add22_out1 = mul35_out1.add((constant416_out1).unsqueeze_dims(&[0isize]));
        let softmax3_out1 = burn::tensor::activation::softmax(add22_out1, 3);
        let matmul15_out1 = softmax3_out1.matmul(transpose27_out1);
        let transpose29_out1 = matmul15_out1.permute([0, 2, 1, 3]);
        let unsqueeze58_out1 = [gather17_out1 as i64];
        let unsqueeze59_out1 = [gather18_out1 as i64];
        let constant419_out1: [i64; 1] = [160i64];
        let concat33_out1: [i64; 3usize] = [
            &unsqueeze58_out1[..],
            &unsqueeze59_out1[..],
            &constant419_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape36_out1 = transpose29_out1.reshape(concat33_out1);
        let linear10_out1 = self.linear10.forward(reshape36_out1);
        let unsqueeze60_out1 = [gather14_out1 as i64];
        let unsqueeze61_out1 = [gather16_out1 as i64];
        let constant421_out1: [i64; 1] = [5i64];
        let constant422_out1: [i64; 1] = [5i64];
        let constant423_out1: [i64; 1] = [14i64];
        let constant424_out1: [i64; 1] = [14i64];
        let concat34_out1: [i64; 6usize] = [
            &unsqueeze60_out1[..],
            &constant421_out1[..],
            &constant422_out1[..],
            &constant423_out1[..],
            &constant424_out1[..],
            &unsqueeze61_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape37_out1 = linear10_out1.reshape(concat34_out1);
        let transpose30_out1 = reshape37_out1.permute([0, 1, 3, 2, 4, 5]);
        let unsqueeze62_out1 = [gather14_out1 as i64];
        let unsqueeze63_out1 = [gather16_out1 as i64];
        let constant427_out1: [i64; 1] = [70i64];
        let constant428_out1: [i64; 1] = [70i64];
        let concat35_out1: [i64; 4usize] = [
            &unsqueeze62_out1[..],
            &constant427_out1[..],
            &constant428_out1[..],
            &unsqueeze63_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape38_out1 = transpose30_out1.reshape(concat35_out1);
        let slice11_out1 = reshape38_out1.slice(s![.., 0..64, .., ..]);
        let slice12_out1 = slice11_out1.slice(s![.., .., 0..64, ..]);
        let unsqueeze64_out1 = [gather14_out1 as i64];
        let unsqueeze65_out1 = [gather15_out1 as i64];
        let unsqueeze66_out1 = [gather16_out1 as i64];
        let concat36_out1: [i64; 3usize] = [
            &unsqueeze64_out1[..],
            &unsqueeze65_out1[..],
            &unsqueeze66_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape39_out1 = slice12_out1.reshape(concat36_out1);
        let add23_out1 = transpose23_out1.add(reshape39_out1);
        let transpose31_out1 = add23_out1.permute([0, 2, 1]);
        let unsqueeze67_out1 = [gather14_out1 as i64];
        let unsqueeze68_out1 = [gather16_out1 as i64];
        let constant444_out1: [i64; 1] = [64i64];
        let constant443_out1: [i64; 1] = [64i64];
        let concat37_out1: [i64; 4usize] = [
            &unsqueeze67_out1[..],
            &unsqueeze68_out1[..],
            &constant443_out1[..],
            &constant444_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape40_out1 = transpose31_out1.reshape(concat37_out1);
        let conv2d17_out1 = self.conv2d17.forward(reshape40_out1);
        let unsqueeze69_out1 = [gather14_out1 as i64];
        let unsqueeze70_out1 = [gather16_out1 as i64];
        let unsqueeze71_out1 = [gather15_out1 as i64];
        let concat38_out1: [i64; 3usize] = [
            &unsqueeze69_out1[..],
            &unsqueeze70_out1[..],
            &unsqueeze71_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape41_out1 = conv2d17_out1.reshape(concat38_out1);
        reshape41_out1
    }
}
#[derive(Module, Debug)]
pub struct Submodule5 {
    layernormalization6: LayerNorm,
    linear11: Linear,
    constant448: burn::module::Param<Tensor<1>>,
    constant449: burn::module::Param<Tensor<1>>,
    constant450: burn::module::Param<Tensor<1>>,
    linear12: Linear,
    layernormalization7: LayerNorm,
    linear13: Linear,
    constant483: burn::module::Param<Tensor<1>>,
    constant484: burn::module::Param<Tensor<3>>,
    linear14: Linear,
    conv2d18: Conv2d,
    #[module(skip)]
    device: Device,
}
impl Submodule5 {
    #[allow(unused_variables)]
    pub fn new(device: &Device) -> Self {
        let layernormalization6 = LayerNormConfig::new(160)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(true)
            .init(device);
        let linear11 = LinearConfig::new(160, 640).with_bias(true).init(device);
        let constant448: burn::module::Param<Tensor<1>> = burn::module::Param::uninitialized(
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
        let constant449: burn::module::Param<Tensor<1>> = burn::module::Param::uninitialized(
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
        let constant450: burn::module::Param<Tensor<1>> = burn::module::Param::uninitialized(
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
        let linear12 = LinearConfig::new(640, 160).with_bias(true).init(device);
        let layernormalization7 = LayerNormConfig::new(160)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(true)
            .init(device);
        let linear13 = LinearConfig::new(160, 480).with_bias(true).init(device);
        let constant483: burn::module::Param<Tensor<1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<
                1,
            >::from_data(
                burn::tensor::TensorData::from([0.1767766922712326f64]),
                (device, burn::tensor::DType::F32),
            ),
            device.clone(),
            false,
            [1].into(),
        );
        let constant484: burn::module::Param<Tensor<3>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<
                3,
            >::zeros([5, 196, 196], (device, burn::tensor::DType::F32)),
            device.clone(),
            false,
            [5, 196, 196].into(),
        );
        let linear14 = LinearConfig::new(160, 160).with_bias(true).init(device);
        let conv2d18 = Conv2dConfig::new([160, 160], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(160)
            .with_bias(true)
            .init(device);
        Self {
            layernormalization6,
            linear11,
            constant448,
            constant449,
            constant450,
            linear12,
            layernormalization7,
            linear13,
            constant483,
            constant484,
            linear14,
            conv2d18,
            device: device.clone(),
        }
    }
    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(&self, reshape41_out1: Tensor<3>) -> Tensor<3> {
        let transpose32_out1 = reshape41_out1.permute([0, 2, 1]);
        let layernormalization6_out1 = {
            let dtype = transpose32_out1.clone().dtype();
            self.layernormalization6
                .forward(transpose32_out1.clone().cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let linear11_out1 = self.linear11.forward(layernormalization6_out1);
        let constant448_out1 = self.constant448.val();
        let div15_out1 = linear11_out1
            .clone()
            .div((constant448_out1).unsqueeze_dims(&[0isize, 1isize]));
        let erf14_out1 = div15_out1.erf();
        let constant449_out1 = self.constant449.val();
        let add24_out1 = erf14_out1
            .add((constant449_out1).unsqueeze_dims(&[0isize, 1isize]));
        let mul36_out1 = linear11_out1.mul(add24_out1);
        let constant450_out1 = self.constant450.val();
        let mul37_out1 = mul36_out1
            .mul((constant450_out1).unsqueeze_dims(&[0isize, 1isize]));
        let linear12_out1 = self.linear12.forward(mul37_out1);
        let add25_out1 = transpose32_out1.add(linear12_out1);
        let shape21_out1: [i64; 3] = {
            let axes = &add25_out1.clone().dims()[0..3];
            let mut output = [0i64; 3];
            for i in 0..3 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let gather19_out1 = shape21_out1[0] as i64;
        let gather20_out1 = shape21_out1[1] as i64;
        let gather21_out1 = shape21_out1[2] as i64;
        let unsqueeze72_out1 = [gather19_out1 as i64];
        let unsqueeze73_out1 = [gather21_out1 as i64];
        let constant455_out1: [i64; 1] = [64i64];
        let constant456_out1: [i64; 1] = [64i64];
        let concat39_out1: [i64; 4usize] = [
            &unsqueeze72_out1[..],
            &constant455_out1[..],
            &constant456_out1[..],
            &unsqueeze73_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape42_out1 = add25_out1.clone().reshape(concat39_out1);
        let constantofshape5_out1: [i64; 2usize] = [0i64, 0i64];
        let constant459_out1: [i64; 6] = [0i64, 0i64, 0i64, 6i64, 0i64, 6i64];
        let concat40_out1: [i64; 8usize] = [
            &constant459_out1[..],
            &constantofshape5_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape43_out1 = {
            let shape_array = concat40_out1 as [i64; 8usize];
            Tensor::<
                1,
                Int,
            >::from_data(
                TensorData::from(shape_array),
                (&self.device, burn::tensor::DType::I64),
            )
        }
            .reshape([-1, 2]);
        let slice13_out1 = {
            let slice_input = reshape43_out1;
            let slice_dims = slice_input.dims();
            let reverse_bounds = |
                dim: usize,
                start: i64,
                end: i64,
            | -> core::ops::Range<usize> {
                if dim == 0 {
                    return 0..0;
                }
                let dim = dim as i64;
                let start = if start < 0 { start.saturating_add(dim) } else { start };
                let end = if end < 0 { end.saturating_add(dim) } else { end };
                let hi = start.clamp(-1, dim - 1) + 1;
                let lo = (end.clamp(-1, dim - 1) + 1).min(hi);
                (lo as usize)..(hi as usize)
            };
            slice_input
                .slice(
                    s![
                        reverse_bounds(slice_dims[0], - 1i64, - 9223372036854775807i64);
                        - 1, ..
                    ],
                )
        };
        let transpose33_out1 = slice13_out1.permute([1, 0]);
        let reshape44_out1 = transpose33_out1.reshape([-1]);
        let pad5_out1 = reshape42_out1
            .pad(
                {
                    let raw: alloc::vec::Vec<i64> = reshape44_out1
                        .to_data()
                        .convert::<i64>()
                        .try_into_vec::<i64>()
                        .unwrap();
                    assert_eq!(
                        raw.len(), 8usize,
                        "Pad: runtime pads length mismatch (expected {}, got {})",
                        8usize, raw.len(),
                    );
                    [
                        (
                            usize::try_from(raw[0usize])
                                .unwrap_or_else(|_| {
                                    panic!(
                                        "Pad: negative pad value {} at index {}", raw[0usize],
                                        0usize
                                    )
                                }),
                            usize::try_from(raw[4usize])
                                .unwrap_or_else(|_| {
                                    panic!(
                                        "Pad: negative pad value {} at index {}", raw[4usize],
                                        4usize
                                    )
                                }),
                        ),
                        (
                            usize::try_from(raw[1usize])
                                .unwrap_or_else(|_| {
                                    panic!(
                                        "Pad: negative pad value {} at index {}", raw[1usize],
                                        1usize
                                    )
                                }),
                            usize::try_from(raw[5usize])
                                .unwrap_or_else(|_| {
                                    panic!(
                                        "Pad: negative pad value {} at index {}", raw[5usize],
                                        5usize
                                    )
                                }),
                        ),
                        (
                            usize::try_from(raw[2usize])
                                .unwrap_or_else(|_| {
                                    panic!(
                                        "Pad: negative pad value {} at index {}", raw[2usize],
                                        2usize
                                    )
                                }),
                            usize::try_from(raw[6usize])
                                .unwrap_or_else(|_| {
                                    panic!(
                                        "Pad: negative pad value {} at index {}", raw[6usize],
                                        6usize
                                    )
                                }),
                        ),
                        (
                            usize::try_from(raw[3usize])
                                .unwrap_or_else(|_| {
                                    panic!(
                                        "Pad: negative pad value {} at index {}", raw[3usize],
                                        3usize
                                    )
                                }),
                            usize::try_from(raw[7usize])
                                .unwrap_or_else(|_| {
                                    panic!(
                                        "Pad: negative pad value {} at index {}", raw[7usize],
                                        7usize
                                    )
                                }),
                        ),
                    ]
                },
                burn::tensor::ops::PadMode::Constant(0f32),
            );
        let unsqueeze74_out1 = [gather19_out1 as i64];
        let unsqueeze75_out1 = [gather21_out1 as i64];
        let constant467_out1: [i64; 1] = [5i64];
        let constant468_out1: [i64; 1] = [14i64];
        let constant469_out1: [i64; 1] = [5i64];
        let constant470_out1: [i64; 1] = [14i64];
        let concat41_out1: [i64; 6usize] = [
            &unsqueeze74_out1[..],
            &constant467_out1[..],
            &constant468_out1[..],
            &constant469_out1[..],
            &constant470_out1[..],
            &unsqueeze75_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape45_out1 = pad5_out1.reshape(concat41_out1);
        let transpose34_out1 = reshape45_out1.permute([0, 1, 3, 2, 4, 5]);
        let constant472_out1 = 5i64;
        let mul38_out1 = gather19_out1 * constant472_out1;
        let constant473_out1 = 5i64;
        let mul39_out1 = mul38_out1 * constant473_out1;
        let unsqueeze76_out1 = [mul39_out1 as i64];
        let unsqueeze77_out1 = [gather21_out1 as i64];
        let constant475_out1: [i64; 1] = [196i64];
        let concat42_out1: [i64; 3usize] = [
            &unsqueeze76_out1[..],
            &constant475_out1[..],
            &unsqueeze77_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape46_out1 = transpose34_out1.reshape(concat42_out1);
        let shape24_out1: [i64; 3] = {
            let axes = &reshape46_out1.clone().dims()[0..3];
            let mut output = [0i64; 3];
            for i in 0..3 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let gather22_out1 = shape24_out1[0] as i64;
        let gather23_out1 = shape24_out1[1] as i64;
        let layernormalization7_out1 = {
            let dtype = reshape46_out1.dtype();
            self.layernormalization7
                .forward(reshape46_out1.cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let linear13_out1 = self.linear13.forward(layernormalization7_out1);
        let unsqueeze78_out1 = [gather22_out1 as i64];
        let unsqueeze79_out1 = [gather23_out1 as i64];
        let constant482_out1: [i64; 1] = [-1i64];
        let constant481_out1: [i64; 1] = [5i64];
        let concat43_out1: [i64; 4usize] = [
            &unsqueeze78_out1[..],
            &unsqueeze79_out1[..],
            &constant481_out1[..],
            &constant482_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape47_out1 = linear13_out1.reshape(concat43_out1);
        let split_tensors = reshape47_out1.split_with_sizes([32, 32, 32].into(), 3);
        let [split4_out1, split4_out2, split4_out3] = split_tensors.try_into().unwrap();
        let transpose35_out1 = split4_out1.permute([0, 2, 1, 3]);
        let transpose36_out1 = split4_out3.permute([0, 2, 1, 3]);
        let transpose37_out1 = split4_out2.permute([0, 2, 3, 1]);
        let matmul20_out1 = transpose35_out1.matmul(transpose37_out1);
        let constant483_out1 = self.constant483.val();
        let mul40_out1 = matmul20_out1
            .mul((constant483_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let constant484_out1 = self.constant484.val();
        let add26_out1 = mul40_out1.add((constant484_out1).unsqueeze_dims(&[0isize]));
        let softmax4_out1 = burn::tensor::activation::softmax(add26_out1, 3);
        let matmul21_out1 = softmax4_out1.matmul(transpose36_out1);
        let transpose38_out1 = matmul21_out1.permute([0, 2, 1, 3]);
        let unsqueeze80_out1 = [gather22_out1 as i64];
        let unsqueeze81_out1 = [gather23_out1 as i64];
        let constant487_out1: [i64; 1] = [160i64];
        let concat44_out1: [i64; 3usize] = [
            &unsqueeze80_out1[..],
            &unsqueeze81_out1[..],
            &constant487_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape48_out1 = transpose38_out1.reshape(concat44_out1);
        let linear14_out1 = self.linear14.forward(reshape48_out1);
        let unsqueeze82_out1 = [gather19_out1 as i64];
        let unsqueeze83_out1 = [gather21_out1 as i64];
        let constant489_out1: [i64; 1] = [5i64];
        let constant490_out1: [i64; 1] = [5i64];
        let constant491_out1: [i64; 1] = [14i64];
        let constant492_out1: [i64; 1] = [14i64];
        let concat45_out1: [i64; 6usize] = [
            &unsqueeze82_out1[..],
            &constant489_out1[..],
            &constant490_out1[..],
            &constant491_out1[..],
            &constant492_out1[..],
            &unsqueeze83_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape49_out1 = linear14_out1.reshape(concat45_out1);
        let transpose39_out1 = reshape49_out1.permute([0, 1, 3, 2, 4, 5]);
        let unsqueeze84_out1 = [gather19_out1 as i64];
        let unsqueeze85_out1 = [gather21_out1 as i64];
        let constant495_out1: [i64; 1] = [70i64];
        let constant496_out1: [i64; 1] = [70i64];
        let concat46_out1: [i64; 4usize] = [
            &unsqueeze84_out1[..],
            &constant495_out1[..],
            &constant496_out1[..],
            &unsqueeze85_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape50_out1 = transpose39_out1.reshape(concat46_out1);
        let slice14_out1 = reshape50_out1.slice(s![.., 0..64, .., ..]);
        let slice15_out1 = slice14_out1.slice(s![.., .., 0..64, ..]);
        let unsqueeze86_out1 = [gather19_out1 as i64];
        let unsqueeze87_out1 = [gather20_out1 as i64];
        let unsqueeze88_out1 = [gather21_out1 as i64];
        let concat47_out1: [i64; 3usize] = [
            &unsqueeze86_out1[..],
            &unsqueeze87_out1[..],
            &unsqueeze88_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape51_out1 = slice15_out1.reshape(concat47_out1);
        let add27_out1 = add25_out1.add(reshape51_out1);
        let transpose40_out1 = add27_out1.permute([0, 2, 1]);
        let unsqueeze89_out1 = [gather19_out1 as i64];
        let unsqueeze90_out1 = [gather21_out1 as i64];
        let constant512_out1: [i64; 1] = [64i64];
        let constant511_out1: [i64; 1] = [64i64];
        let concat48_out1: [i64; 4usize] = [
            &unsqueeze89_out1[..],
            &unsqueeze90_out1[..],
            &constant511_out1[..],
            &constant512_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape52_out1 = transpose40_out1.reshape(concat48_out1);
        let conv2d18_out1 = self.conv2d18.forward(reshape52_out1);
        let unsqueeze91_out1 = [gather19_out1 as i64];
        let unsqueeze92_out1 = [gather21_out1 as i64];
        let unsqueeze93_out1 = [gather20_out1 as i64];
        let concat49_out1: [i64; 3usize] = [
            &unsqueeze91_out1[..],
            &unsqueeze92_out1[..],
            &unsqueeze93_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape53_out1 = conv2d18_out1.reshape(concat49_out1);
        reshape53_out1
    }
}
#[derive(Module, Debug)]
pub struct Submodule6 {
    layernormalization8: LayerNorm,
    linear15: Linear,
    constant516: burn::module::Param<Tensor<1>>,
    constant517: burn::module::Param<Tensor<1>>,
    constant518: burn::module::Param<Tensor<1>>,
    linear16: Linear,
    layernormalization9: LayerNorm,
    linear17: Linear,
    constant551: burn::module::Param<Tensor<1>>,
    constant552: burn::module::Param<Tensor<3>>,
    linear18: Linear,
    conv2d19: Conv2d,
    #[module(skip)]
    device: Device,
}
impl Submodule6 {
    #[allow(unused_variables)]
    pub fn new(device: &Device) -> Self {
        let layernormalization8 = LayerNormConfig::new(160)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(true)
            .init(device);
        let linear15 = LinearConfig::new(160, 640).with_bias(true).init(device);
        let constant516: burn::module::Param<Tensor<1>> = burn::module::Param::uninitialized(
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
        let constant517: burn::module::Param<Tensor<1>> = burn::module::Param::uninitialized(
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
        let constant518: burn::module::Param<Tensor<1>> = burn::module::Param::uninitialized(
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
        let linear16 = LinearConfig::new(640, 160).with_bias(true).init(device);
        let layernormalization9 = LayerNormConfig::new(160)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(true)
            .init(device);
        let linear17 = LinearConfig::new(160, 480).with_bias(true).init(device);
        let constant551: burn::module::Param<Tensor<1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<
                1,
            >::from_data(
                burn::tensor::TensorData::from([0.1767766922712326f64]),
                (device, burn::tensor::DType::F32),
            ),
            device.clone(),
            false,
            [1].into(),
        );
        let constant552: burn::module::Param<Tensor<3>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<
                3,
            >::zeros([5, 196, 196], (device, burn::tensor::DType::F32)),
            device.clone(),
            false,
            [5, 196, 196].into(),
        );
        let linear18 = LinearConfig::new(160, 160).with_bias(true).init(device);
        let conv2d19 = Conv2dConfig::new([160, 160], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(160)
            .with_bias(true)
            .init(device);
        Self {
            layernormalization8,
            linear15,
            constant516,
            constant517,
            constant518,
            linear16,
            layernormalization9,
            linear17,
            constant551,
            constant552,
            linear18,
            conv2d19,
            device: device.clone(),
        }
    }
    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(&self, reshape53_out1: Tensor<3>) -> Tensor<3> {
        let transpose41_out1 = reshape53_out1.permute([0, 2, 1]);
        let layernormalization8_out1 = {
            let dtype = transpose41_out1.clone().dtype();
            self.layernormalization8
                .forward(transpose41_out1.clone().cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let linear15_out1 = self.linear15.forward(layernormalization8_out1);
        let constant516_out1 = self.constant516.val();
        let div16_out1 = linear15_out1
            .clone()
            .div((constant516_out1).unsqueeze_dims(&[0isize, 1isize]));
        let erf15_out1 = div16_out1.erf();
        let constant517_out1 = self.constant517.val();
        let add28_out1 = erf15_out1
            .add((constant517_out1).unsqueeze_dims(&[0isize, 1isize]));
        let mul41_out1 = linear15_out1.mul(add28_out1);
        let constant518_out1 = self.constant518.val();
        let mul42_out1 = mul41_out1
            .mul((constant518_out1).unsqueeze_dims(&[0isize, 1isize]));
        let linear16_out1 = self.linear16.forward(mul42_out1);
        let add29_out1 = transpose41_out1.add(linear16_out1);
        let shape26_out1: [i64; 3] = {
            let axes = &add29_out1.clone().dims()[0..3];
            let mut output = [0i64; 3];
            for i in 0..3 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let gather24_out1 = shape26_out1[0] as i64;
        let gather25_out1 = shape26_out1[1] as i64;
        let gather26_out1 = shape26_out1[2] as i64;
        let unsqueeze94_out1 = [gather24_out1 as i64];
        let unsqueeze95_out1 = [gather26_out1 as i64];
        let constant523_out1: [i64; 1] = [64i64];
        let constant524_out1: [i64; 1] = [64i64];
        let concat50_out1: [i64; 4usize] = [
            &unsqueeze94_out1[..],
            &constant523_out1[..],
            &constant524_out1[..],
            &unsqueeze95_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape54_out1 = add29_out1.clone().reshape(concat50_out1);
        let constantofshape6_out1: [i64; 2usize] = [0i64, 0i64];
        let constant527_out1: [i64; 6] = [0i64, 0i64, 0i64, 6i64, 0i64, 6i64];
        let concat51_out1: [i64; 8usize] = [
            &constant527_out1[..],
            &constantofshape6_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape55_out1 = {
            let shape_array = concat51_out1 as [i64; 8usize];
            Tensor::<
                1,
                Int,
            >::from_data(
                TensorData::from(shape_array),
                (&self.device, burn::tensor::DType::I64),
            )
        }
            .reshape([-1, 2]);
        let slice16_out1 = {
            let slice_input = reshape55_out1;
            let slice_dims = slice_input.dims();
            let reverse_bounds = |
                dim: usize,
                start: i64,
                end: i64,
            | -> core::ops::Range<usize> {
                if dim == 0 {
                    return 0..0;
                }
                let dim = dim as i64;
                let start = if start < 0 { start.saturating_add(dim) } else { start };
                let end = if end < 0 { end.saturating_add(dim) } else { end };
                let hi = start.clamp(-1, dim - 1) + 1;
                let lo = (end.clamp(-1, dim - 1) + 1).min(hi);
                (lo as usize)..(hi as usize)
            };
            slice_input
                .slice(
                    s![
                        reverse_bounds(slice_dims[0], - 1i64, - 9223372036854775807i64);
                        - 1, ..
                    ],
                )
        };
        let transpose42_out1 = slice16_out1.permute([1, 0]);
        let reshape56_out1 = transpose42_out1.reshape([-1]);
        let pad6_out1 = reshape54_out1
            .pad(
                {
                    let raw: alloc::vec::Vec<i64> = reshape56_out1
                        .to_data()
                        .convert::<i64>()
                        .try_into_vec::<i64>()
                        .unwrap();
                    assert_eq!(
                        raw.len(), 8usize,
                        "Pad: runtime pads length mismatch (expected {}, got {})",
                        8usize, raw.len(),
                    );
                    [
                        (
                            usize::try_from(raw[0usize])
                                .unwrap_or_else(|_| {
                                    panic!(
                                        "Pad: negative pad value {} at index {}", raw[0usize],
                                        0usize
                                    )
                                }),
                            usize::try_from(raw[4usize])
                                .unwrap_or_else(|_| {
                                    panic!(
                                        "Pad: negative pad value {} at index {}", raw[4usize],
                                        4usize
                                    )
                                }),
                        ),
                        (
                            usize::try_from(raw[1usize])
                                .unwrap_or_else(|_| {
                                    panic!(
                                        "Pad: negative pad value {} at index {}", raw[1usize],
                                        1usize
                                    )
                                }),
                            usize::try_from(raw[5usize])
                                .unwrap_or_else(|_| {
                                    panic!(
                                        "Pad: negative pad value {} at index {}", raw[5usize],
                                        5usize
                                    )
                                }),
                        ),
                        (
                            usize::try_from(raw[2usize])
                                .unwrap_or_else(|_| {
                                    panic!(
                                        "Pad: negative pad value {} at index {}", raw[2usize],
                                        2usize
                                    )
                                }),
                            usize::try_from(raw[6usize])
                                .unwrap_or_else(|_| {
                                    panic!(
                                        "Pad: negative pad value {} at index {}", raw[6usize],
                                        6usize
                                    )
                                }),
                        ),
                        (
                            usize::try_from(raw[3usize])
                                .unwrap_or_else(|_| {
                                    panic!(
                                        "Pad: negative pad value {} at index {}", raw[3usize],
                                        3usize
                                    )
                                }),
                            usize::try_from(raw[7usize])
                                .unwrap_or_else(|_| {
                                    panic!(
                                        "Pad: negative pad value {} at index {}", raw[7usize],
                                        7usize
                                    )
                                }),
                        ),
                    ]
                },
                burn::tensor::ops::PadMode::Constant(0f32),
            );
        let unsqueeze96_out1 = [gather24_out1 as i64];
        let unsqueeze97_out1 = [gather26_out1 as i64];
        let constant535_out1: [i64; 1] = [5i64];
        let constant536_out1: [i64; 1] = [14i64];
        let constant537_out1: [i64; 1] = [5i64];
        let constant538_out1: [i64; 1] = [14i64];
        let concat52_out1: [i64; 6usize] = [
            &unsqueeze96_out1[..],
            &constant535_out1[..],
            &constant536_out1[..],
            &constant537_out1[..],
            &constant538_out1[..],
            &unsqueeze97_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape57_out1 = pad6_out1.reshape(concat52_out1);
        let transpose43_out1 = reshape57_out1.permute([0, 1, 3, 2, 4, 5]);
        let constant540_out1 = 5i64;
        let mul43_out1 = gather24_out1 * constant540_out1;
        let constant541_out1 = 5i64;
        let mul44_out1 = mul43_out1 * constant541_out1;
        let unsqueeze98_out1 = [mul44_out1 as i64];
        let unsqueeze99_out1 = [gather26_out1 as i64];
        let constant543_out1: [i64; 1] = [196i64];
        let concat53_out1: [i64; 3usize] = [
            &unsqueeze98_out1[..],
            &constant543_out1[..],
            &unsqueeze99_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape58_out1 = transpose43_out1.reshape(concat53_out1);
        let shape29_out1: [i64; 3] = {
            let axes = &reshape58_out1.clone().dims()[0..3];
            let mut output = [0i64; 3];
            for i in 0..3 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let gather27_out1 = shape29_out1[0] as i64;
        let gather28_out1 = shape29_out1[1] as i64;
        let layernormalization9_out1 = {
            let dtype = reshape58_out1.dtype();
            self.layernormalization9
                .forward(reshape58_out1.cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let linear17_out1 = self.linear17.forward(layernormalization9_out1);
        let unsqueeze100_out1 = [gather27_out1 as i64];
        let unsqueeze101_out1 = [gather28_out1 as i64];
        let constant550_out1: [i64; 1] = [-1i64];
        let constant549_out1: [i64; 1] = [5i64];
        let concat54_out1: [i64; 4usize] = [
            &unsqueeze100_out1[..],
            &unsqueeze101_out1[..],
            &constant549_out1[..],
            &constant550_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape59_out1 = linear17_out1.reshape(concat54_out1);
        let split_tensors = reshape59_out1.split_with_sizes([32, 32, 32].into(), 3);
        let [split5_out1, split5_out2, split5_out3] = split_tensors.try_into().unwrap();
        let transpose44_out1 = split5_out1.permute([0, 2, 1, 3]);
        let transpose45_out1 = split5_out3.permute([0, 2, 1, 3]);
        let transpose46_out1 = split5_out2.permute([0, 2, 3, 1]);
        let matmul26_out1 = transpose44_out1.matmul(transpose46_out1);
        let constant551_out1 = self.constant551.val();
        let mul45_out1 = matmul26_out1
            .mul((constant551_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let constant552_out1 = self.constant552.val();
        let add30_out1 = mul45_out1.add((constant552_out1).unsqueeze_dims(&[0isize]));
        let softmax5_out1 = burn::tensor::activation::softmax(add30_out1, 3);
        let matmul27_out1 = softmax5_out1.matmul(transpose45_out1);
        let transpose47_out1 = matmul27_out1.permute([0, 2, 1, 3]);
        let unsqueeze102_out1 = [gather27_out1 as i64];
        let unsqueeze103_out1 = [gather28_out1 as i64];
        let constant555_out1: [i64; 1] = [160i64];
        let concat55_out1: [i64; 3usize] = [
            &unsqueeze102_out1[..],
            &unsqueeze103_out1[..],
            &constant555_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape60_out1 = transpose47_out1.reshape(concat55_out1);
        let linear18_out1 = self.linear18.forward(reshape60_out1);
        let unsqueeze104_out1 = [gather24_out1 as i64];
        let unsqueeze105_out1 = [gather26_out1 as i64];
        let constant557_out1: [i64; 1] = [5i64];
        let constant558_out1: [i64; 1] = [5i64];
        let constant559_out1: [i64; 1] = [14i64];
        let constant560_out1: [i64; 1] = [14i64];
        let concat56_out1: [i64; 6usize] = [
            &unsqueeze104_out1[..],
            &constant557_out1[..],
            &constant558_out1[..],
            &constant559_out1[..],
            &constant560_out1[..],
            &unsqueeze105_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape61_out1 = linear18_out1.reshape(concat56_out1);
        let transpose48_out1 = reshape61_out1.permute([0, 1, 3, 2, 4, 5]);
        let unsqueeze106_out1 = [gather24_out1 as i64];
        let unsqueeze107_out1 = [gather26_out1 as i64];
        let constant563_out1: [i64; 1] = [70i64];
        let constant564_out1: [i64; 1] = [70i64];
        let concat57_out1: [i64; 4usize] = [
            &unsqueeze106_out1[..],
            &constant563_out1[..],
            &constant564_out1[..],
            &unsqueeze107_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape62_out1 = transpose48_out1.reshape(concat57_out1);
        let slice17_out1 = reshape62_out1.slice(s![.., 0..64, .., ..]);
        let slice18_out1 = slice17_out1.slice(s![.., .., 0..64, ..]);
        let unsqueeze108_out1 = [gather24_out1 as i64];
        let unsqueeze109_out1 = [gather25_out1 as i64];
        let unsqueeze110_out1 = [gather26_out1 as i64];
        let concat58_out1: [i64; 3usize] = [
            &unsqueeze108_out1[..],
            &unsqueeze109_out1[..],
            &unsqueeze110_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape63_out1 = slice18_out1.reshape(concat58_out1);
        let add31_out1 = add29_out1.add(reshape63_out1);
        let transpose49_out1 = add31_out1.permute([0, 2, 1]);
        let unsqueeze111_out1 = [gather24_out1 as i64];
        let unsqueeze112_out1 = [gather26_out1 as i64];
        let constant580_out1: [i64; 1] = [64i64];
        let constant579_out1: [i64; 1] = [64i64];
        let concat59_out1: [i64; 4usize] = [
            &unsqueeze111_out1[..],
            &unsqueeze112_out1[..],
            &constant579_out1[..],
            &constant580_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape64_out1 = transpose49_out1.reshape(concat59_out1);
        let conv2d19_out1 = self.conv2d19.forward(reshape64_out1);
        let unsqueeze113_out1 = [gather24_out1 as i64];
        let unsqueeze114_out1 = [gather26_out1 as i64];
        let unsqueeze115_out1 = [gather25_out1 as i64];
        let concat60_out1: [i64; 3usize] = [
            &unsqueeze113_out1[..],
            &unsqueeze114_out1[..],
            &unsqueeze115_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape65_out1 = conv2d19_out1.reshape(concat60_out1);
        reshape65_out1
    }
}
#[derive(Module, Debug)]
pub struct Submodule7 {
    layernormalization10: LayerNorm,
    linear19: Linear,
    constant584: burn::module::Param<Tensor<1>>,
    constant585: burn::module::Param<Tensor<1>>,
    constant586: burn::module::Param<Tensor<1>>,
    linear20: Linear,
    layernormalization11: LayerNorm,
    linear21: Linear,
    constant619: burn::module::Param<Tensor<1>>,
    constant620: burn::module::Param<Tensor<3>>,
    linear22: Linear,
    conv2d20: Conv2d,
    #[module(skip)]
    device: Device,
}
impl Submodule7 {
    #[allow(unused_variables)]
    pub fn new(device: &Device) -> Self {
        let layernormalization10 = LayerNormConfig::new(160)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(true)
            .init(device);
        let linear19 = LinearConfig::new(160, 640).with_bias(true).init(device);
        let constant584: burn::module::Param<Tensor<1>> = burn::module::Param::uninitialized(
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
        let constant585: burn::module::Param<Tensor<1>> = burn::module::Param::uninitialized(
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
        let constant586: burn::module::Param<Tensor<1>> = burn::module::Param::uninitialized(
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
        let linear20 = LinearConfig::new(640, 160).with_bias(true).init(device);
        let layernormalization11 = LayerNormConfig::new(160)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(true)
            .init(device);
        let linear21 = LinearConfig::new(160, 480).with_bias(true).init(device);
        let constant619: burn::module::Param<Tensor<1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<
                1,
            >::from_data(
                burn::tensor::TensorData::from([0.1767766922712326f64]),
                (device, burn::tensor::DType::F32),
            ),
            device.clone(),
            false,
            [1].into(),
        );
        let constant620: burn::module::Param<Tensor<3>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<
                3,
            >::zeros([5, 196, 196], (device, burn::tensor::DType::F32)),
            device.clone(),
            false,
            [5, 196, 196].into(),
        );
        let linear22 = LinearConfig::new(160, 160).with_bias(true).init(device);
        let conv2d20 = Conv2dConfig::new([160, 160], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(160)
            .with_bias(true)
            .init(device);
        Self {
            layernormalization10,
            linear19,
            constant584,
            constant585,
            constant586,
            linear20,
            layernormalization11,
            linear21,
            constant619,
            constant620,
            linear22,
            conv2d20,
            device: device.clone(),
        }
    }
    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(&self, reshape65_out1: Tensor<3>) -> Tensor<3> {
        let transpose50_out1 = reshape65_out1.permute([0, 2, 1]);
        let layernormalization10_out1 = {
            let dtype = transpose50_out1.clone().dtype();
            self.layernormalization10
                .forward(transpose50_out1.clone().cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let linear19_out1 = self.linear19.forward(layernormalization10_out1);
        let constant584_out1 = self.constant584.val();
        let div17_out1 = linear19_out1
            .clone()
            .div((constant584_out1).unsqueeze_dims(&[0isize, 1isize]));
        let erf16_out1 = div17_out1.erf();
        let constant585_out1 = self.constant585.val();
        let add32_out1 = erf16_out1
            .add((constant585_out1).unsqueeze_dims(&[0isize, 1isize]));
        let mul46_out1 = linear19_out1.mul(add32_out1);
        let constant586_out1 = self.constant586.val();
        let mul47_out1 = mul46_out1
            .mul((constant586_out1).unsqueeze_dims(&[0isize, 1isize]));
        let linear20_out1 = self.linear20.forward(mul47_out1);
        let add33_out1 = transpose50_out1.add(linear20_out1);
        let shape31_out1: [i64; 3] = {
            let axes = &add33_out1.clone().dims()[0..3];
            let mut output = [0i64; 3];
            for i in 0..3 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let gather29_out1 = shape31_out1[0] as i64;
        let gather30_out1 = shape31_out1[1] as i64;
        let gather31_out1 = shape31_out1[2] as i64;
        let unsqueeze116_out1 = [gather29_out1 as i64];
        let unsqueeze117_out1 = [gather31_out1 as i64];
        let constant591_out1: [i64; 1] = [64i64];
        let constant592_out1: [i64; 1] = [64i64];
        let concat61_out1: [i64; 4usize] = [
            &unsqueeze116_out1[..],
            &constant591_out1[..],
            &constant592_out1[..],
            &unsqueeze117_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape66_out1 = add33_out1.clone().reshape(concat61_out1);
        let constantofshape7_out1: [i64; 2usize] = [0i64, 0i64];
        let constant595_out1: [i64; 6] = [0i64, 0i64, 0i64, 6i64, 0i64, 6i64];
        let concat62_out1: [i64; 8usize] = [
            &constant595_out1[..],
            &constantofshape7_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape67_out1 = {
            let shape_array = concat62_out1 as [i64; 8usize];
            Tensor::<
                1,
                Int,
            >::from_data(
                TensorData::from(shape_array),
                (&self.device, burn::tensor::DType::I64),
            )
        }
            .reshape([-1, 2]);
        let slice19_out1 = {
            let slice_input = reshape67_out1;
            let slice_dims = slice_input.dims();
            let reverse_bounds = |
                dim: usize,
                start: i64,
                end: i64,
            | -> core::ops::Range<usize> {
                if dim == 0 {
                    return 0..0;
                }
                let dim = dim as i64;
                let start = if start < 0 { start.saturating_add(dim) } else { start };
                let end = if end < 0 { end.saturating_add(dim) } else { end };
                let hi = start.clamp(-1, dim - 1) + 1;
                let lo = (end.clamp(-1, dim - 1) + 1).min(hi);
                (lo as usize)..(hi as usize)
            };
            slice_input
                .slice(
                    s![
                        reverse_bounds(slice_dims[0], - 1i64, - 9223372036854775807i64);
                        - 1, ..
                    ],
                )
        };
        let transpose51_out1 = slice19_out1.permute([1, 0]);
        let reshape68_out1 = transpose51_out1.reshape([-1]);
        let pad7_out1 = reshape66_out1
            .pad(
                {
                    let raw: alloc::vec::Vec<i64> = reshape68_out1
                        .to_data()
                        .convert::<i64>()
                        .try_into_vec::<i64>()
                        .unwrap();
                    assert_eq!(
                        raw.len(), 8usize,
                        "Pad: runtime pads length mismatch (expected {}, got {})",
                        8usize, raw.len(),
                    );
                    [
                        (
                            usize::try_from(raw[0usize])
                                .unwrap_or_else(|_| {
                                    panic!(
                                        "Pad: negative pad value {} at index {}", raw[0usize],
                                        0usize
                                    )
                                }),
                            usize::try_from(raw[4usize])
                                .unwrap_or_else(|_| {
                                    panic!(
                                        "Pad: negative pad value {} at index {}", raw[4usize],
                                        4usize
                                    )
                                }),
                        ),
                        (
                            usize::try_from(raw[1usize])
                                .unwrap_or_else(|_| {
                                    panic!(
                                        "Pad: negative pad value {} at index {}", raw[1usize],
                                        1usize
                                    )
                                }),
                            usize::try_from(raw[5usize])
                                .unwrap_or_else(|_| {
                                    panic!(
                                        "Pad: negative pad value {} at index {}", raw[5usize],
                                        5usize
                                    )
                                }),
                        ),
                        (
                            usize::try_from(raw[2usize])
                                .unwrap_or_else(|_| {
                                    panic!(
                                        "Pad: negative pad value {} at index {}", raw[2usize],
                                        2usize
                                    )
                                }),
                            usize::try_from(raw[6usize])
                                .unwrap_or_else(|_| {
                                    panic!(
                                        "Pad: negative pad value {} at index {}", raw[6usize],
                                        6usize
                                    )
                                }),
                        ),
                        (
                            usize::try_from(raw[3usize])
                                .unwrap_or_else(|_| {
                                    panic!(
                                        "Pad: negative pad value {} at index {}", raw[3usize],
                                        3usize
                                    )
                                }),
                            usize::try_from(raw[7usize])
                                .unwrap_or_else(|_| {
                                    panic!(
                                        "Pad: negative pad value {} at index {}", raw[7usize],
                                        7usize
                                    )
                                }),
                        ),
                    ]
                },
                burn::tensor::ops::PadMode::Constant(0f32),
            );
        let unsqueeze118_out1 = [gather29_out1 as i64];
        let unsqueeze119_out1 = [gather31_out1 as i64];
        let constant603_out1: [i64; 1] = [5i64];
        let constant604_out1: [i64; 1] = [14i64];
        let constant605_out1: [i64; 1] = [5i64];
        let constant606_out1: [i64; 1] = [14i64];
        let concat63_out1: [i64; 6usize] = [
            &unsqueeze118_out1[..],
            &constant603_out1[..],
            &constant604_out1[..],
            &constant605_out1[..],
            &constant606_out1[..],
            &unsqueeze119_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape69_out1 = pad7_out1.reshape(concat63_out1);
        let transpose52_out1 = reshape69_out1.permute([0, 1, 3, 2, 4, 5]);
        let constant608_out1 = 5i64;
        let mul48_out1 = gather29_out1 * constant608_out1;
        let constant609_out1 = 5i64;
        let mul49_out1 = mul48_out1 * constant609_out1;
        let unsqueeze120_out1 = [mul49_out1 as i64];
        let unsqueeze121_out1 = [gather31_out1 as i64];
        let constant611_out1: [i64; 1] = [196i64];
        let concat64_out1: [i64; 3usize] = [
            &unsqueeze120_out1[..],
            &constant611_out1[..],
            &unsqueeze121_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape70_out1 = transpose52_out1.reshape(concat64_out1);
        let shape34_out1: [i64; 3] = {
            let axes = &reshape70_out1.clone().dims()[0..3];
            let mut output = [0i64; 3];
            for i in 0..3 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let gather32_out1 = shape34_out1[0] as i64;
        let gather33_out1 = shape34_out1[1] as i64;
        let layernormalization11_out1 = {
            let dtype = reshape70_out1.dtype();
            self.layernormalization11
                .forward(reshape70_out1.cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let linear21_out1 = self.linear21.forward(layernormalization11_out1);
        let unsqueeze122_out1 = [gather32_out1 as i64];
        let unsqueeze123_out1 = [gather33_out1 as i64];
        let constant618_out1: [i64; 1] = [-1i64];
        let constant617_out1: [i64; 1] = [5i64];
        let concat65_out1: [i64; 4usize] = [
            &unsqueeze122_out1[..],
            &unsqueeze123_out1[..],
            &constant617_out1[..],
            &constant618_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape71_out1 = linear21_out1.reshape(concat65_out1);
        let split_tensors = reshape71_out1.split_with_sizes([32, 32, 32].into(), 3);
        let [split6_out1, split6_out2, split6_out3] = split_tensors.try_into().unwrap();
        let transpose53_out1 = split6_out1.permute([0, 2, 1, 3]);
        let transpose54_out1 = split6_out3.permute([0, 2, 1, 3]);
        let transpose55_out1 = split6_out2.permute([0, 2, 3, 1]);
        let matmul32_out1 = transpose53_out1.matmul(transpose55_out1);
        let constant619_out1 = self.constant619.val();
        let mul50_out1 = matmul32_out1
            .mul((constant619_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let constant620_out1 = self.constant620.val();
        let add34_out1 = mul50_out1.add((constant620_out1).unsqueeze_dims(&[0isize]));
        let softmax6_out1 = burn::tensor::activation::softmax(add34_out1, 3);
        let matmul33_out1 = softmax6_out1.matmul(transpose54_out1);
        let transpose56_out1 = matmul33_out1.permute([0, 2, 1, 3]);
        let unsqueeze124_out1 = [gather32_out1 as i64];
        let unsqueeze125_out1 = [gather33_out1 as i64];
        let constant623_out1: [i64; 1] = [160i64];
        let concat66_out1: [i64; 3usize] = [
            &unsqueeze124_out1[..],
            &unsqueeze125_out1[..],
            &constant623_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape72_out1 = transpose56_out1.reshape(concat66_out1);
        let linear22_out1 = self.linear22.forward(reshape72_out1);
        let unsqueeze126_out1 = [gather29_out1 as i64];
        let unsqueeze127_out1 = [gather31_out1 as i64];
        let constant625_out1: [i64; 1] = [5i64];
        let constant626_out1: [i64; 1] = [5i64];
        let constant627_out1: [i64; 1] = [14i64];
        let constant628_out1: [i64; 1] = [14i64];
        let concat67_out1: [i64; 6usize] = [
            &unsqueeze126_out1[..],
            &constant625_out1[..],
            &constant626_out1[..],
            &constant627_out1[..],
            &constant628_out1[..],
            &unsqueeze127_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape73_out1 = linear22_out1.reshape(concat67_out1);
        let transpose57_out1 = reshape73_out1.permute([0, 1, 3, 2, 4, 5]);
        let unsqueeze128_out1 = [gather29_out1 as i64];
        let unsqueeze129_out1 = [gather31_out1 as i64];
        let constant631_out1: [i64; 1] = [70i64];
        let constant632_out1: [i64; 1] = [70i64];
        let concat68_out1: [i64; 4usize] = [
            &unsqueeze128_out1[..],
            &constant631_out1[..],
            &constant632_out1[..],
            &unsqueeze129_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape74_out1 = transpose57_out1.reshape(concat68_out1);
        let slice20_out1 = reshape74_out1.slice(s![.., 0..64, .., ..]);
        let slice21_out1 = slice20_out1.slice(s![.., .., 0..64, ..]);
        let unsqueeze130_out1 = [gather29_out1 as i64];
        let unsqueeze131_out1 = [gather30_out1 as i64];
        let unsqueeze132_out1 = [gather31_out1 as i64];
        let concat69_out1: [i64; 3usize] = [
            &unsqueeze130_out1[..],
            &unsqueeze131_out1[..],
            &unsqueeze132_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape75_out1 = slice21_out1.reshape(concat69_out1);
        let add35_out1 = add33_out1.add(reshape75_out1);
        let transpose58_out1 = add35_out1.permute([0, 2, 1]);
        let unsqueeze133_out1 = [gather29_out1 as i64];
        let unsqueeze134_out1 = [gather31_out1 as i64];
        let constant648_out1: [i64; 1] = [64i64];
        let constant647_out1: [i64; 1] = [64i64];
        let concat70_out1: [i64; 4usize] = [
            &unsqueeze133_out1[..],
            &unsqueeze134_out1[..],
            &constant647_out1[..],
            &constant648_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape76_out1 = transpose58_out1.reshape(concat70_out1);
        let conv2d20_out1 = self.conv2d20.forward(reshape76_out1);
        let unsqueeze135_out1 = [gather29_out1 as i64];
        let unsqueeze136_out1 = [gather31_out1 as i64];
        let unsqueeze137_out1 = [gather30_out1 as i64];
        let concat71_out1: [i64; 3usize] = [
            &unsqueeze135_out1[..],
            &unsqueeze136_out1[..],
            &unsqueeze137_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape77_out1 = conv2d20_out1.reshape(concat71_out1);
        reshape77_out1
    }
}
#[derive(Module, Debug)]
pub struct Submodule8 {
    layernormalization12: LayerNorm,
    linear23: Linear,
    constant652: burn::module::Param<Tensor<1>>,
    constant653: burn::module::Param<Tensor<1>>,
    constant654: burn::module::Param<Tensor<1>>,
    linear24: Linear,
    layernormalization13: LayerNorm,
    linear25: Linear,
    constant687: burn::module::Param<Tensor<1>>,
    constant688: burn::module::Param<Tensor<3>>,
    linear26: Linear,
    conv2d21: Conv2d,
    #[module(skip)]
    device: Device,
}
impl Submodule8 {
    #[allow(unused_variables)]
    pub fn new(device: &Device) -> Self {
        let layernormalization12 = LayerNormConfig::new(160)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(true)
            .init(device);
        let linear23 = LinearConfig::new(160, 640).with_bias(true).init(device);
        let constant652: burn::module::Param<Tensor<1>> = burn::module::Param::uninitialized(
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
        let constant653: burn::module::Param<Tensor<1>> = burn::module::Param::uninitialized(
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
        let constant654: burn::module::Param<Tensor<1>> = burn::module::Param::uninitialized(
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
        let linear24 = LinearConfig::new(640, 160).with_bias(true).init(device);
        let layernormalization13 = LayerNormConfig::new(160)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(true)
            .init(device);
        let linear25 = LinearConfig::new(160, 480).with_bias(true).init(device);
        let constant687: burn::module::Param<Tensor<1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<
                1,
            >::from_data(
                burn::tensor::TensorData::from([0.1767766922712326f64]),
                (device, burn::tensor::DType::F32),
            ),
            device.clone(),
            false,
            [1].into(),
        );
        let constant688: burn::module::Param<Tensor<3>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<
                3,
            >::zeros([5, 196, 196], (device, burn::tensor::DType::F32)),
            device.clone(),
            false,
            [5, 196, 196].into(),
        );
        let linear26 = LinearConfig::new(160, 160).with_bias(true).init(device);
        let conv2d21 = Conv2dConfig::new([160, 160], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(160)
            .with_bias(true)
            .init(device);
        Self {
            layernormalization12,
            linear23,
            constant652,
            constant653,
            constant654,
            linear24,
            layernormalization13,
            linear25,
            constant687,
            constant688,
            linear26,
            conv2d21,
            device: device.clone(),
        }
    }
    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(&self, reshape77_out1: Tensor<3>) -> Tensor<3> {
        let transpose59_out1 = reshape77_out1.permute([0, 2, 1]);
        let layernormalization12_out1 = {
            let dtype = transpose59_out1.clone().dtype();
            self.layernormalization12
                .forward(transpose59_out1.clone().cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let linear23_out1 = self.linear23.forward(layernormalization12_out1);
        let constant652_out1 = self.constant652.val();
        let div18_out1 = linear23_out1
            .clone()
            .div((constant652_out1).unsqueeze_dims(&[0isize, 1isize]));
        let erf17_out1 = div18_out1.erf();
        let constant653_out1 = self.constant653.val();
        let add36_out1 = erf17_out1
            .add((constant653_out1).unsqueeze_dims(&[0isize, 1isize]));
        let mul51_out1 = linear23_out1.mul(add36_out1);
        let constant654_out1 = self.constant654.val();
        let mul52_out1 = mul51_out1
            .mul((constant654_out1).unsqueeze_dims(&[0isize, 1isize]));
        let linear24_out1 = self.linear24.forward(mul52_out1);
        let add37_out1 = transpose59_out1.add(linear24_out1);
        let shape36_out1: [i64; 3] = {
            let axes = &add37_out1.clone().dims()[0..3];
            let mut output = [0i64; 3];
            for i in 0..3 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let gather34_out1 = shape36_out1[0] as i64;
        let gather35_out1 = shape36_out1[1] as i64;
        let gather36_out1 = shape36_out1[2] as i64;
        let unsqueeze138_out1 = [gather34_out1 as i64];
        let unsqueeze139_out1 = [gather36_out1 as i64];
        let constant659_out1: [i64; 1] = [64i64];
        let constant660_out1: [i64; 1] = [64i64];
        let concat72_out1: [i64; 4usize] = [
            &unsqueeze138_out1[..],
            &constant659_out1[..],
            &constant660_out1[..],
            &unsqueeze139_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape78_out1 = add37_out1.clone().reshape(concat72_out1);
        let constantofshape8_out1: [i64; 2usize] = [0i64, 0i64];
        let constant663_out1: [i64; 6] = [0i64, 0i64, 0i64, 6i64, 0i64, 6i64];
        let concat73_out1: [i64; 8usize] = [
            &constant663_out1[..],
            &constantofshape8_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape79_out1 = {
            let shape_array = concat73_out1 as [i64; 8usize];
            Tensor::<
                1,
                Int,
            >::from_data(
                TensorData::from(shape_array),
                (&self.device, burn::tensor::DType::I64),
            )
        }
            .reshape([-1, 2]);
        let slice22_out1 = {
            let slice_input = reshape79_out1;
            let slice_dims = slice_input.dims();
            let reverse_bounds = |
                dim: usize,
                start: i64,
                end: i64,
            | -> core::ops::Range<usize> {
                if dim == 0 {
                    return 0..0;
                }
                let dim = dim as i64;
                let start = if start < 0 { start.saturating_add(dim) } else { start };
                let end = if end < 0 { end.saturating_add(dim) } else { end };
                let hi = start.clamp(-1, dim - 1) + 1;
                let lo = (end.clamp(-1, dim - 1) + 1).min(hi);
                (lo as usize)..(hi as usize)
            };
            slice_input
                .slice(
                    s![
                        reverse_bounds(slice_dims[0], - 1i64, - 9223372036854775807i64);
                        - 1, ..
                    ],
                )
        };
        let transpose60_out1 = slice22_out1.permute([1, 0]);
        let reshape80_out1 = transpose60_out1.reshape([-1]);
        let pad8_out1 = reshape78_out1
            .pad(
                {
                    let raw: alloc::vec::Vec<i64> = reshape80_out1
                        .to_data()
                        .convert::<i64>()
                        .try_into_vec::<i64>()
                        .unwrap();
                    assert_eq!(
                        raw.len(), 8usize,
                        "Pad: runtime pads length mismatch (expected {}, got {})",
                        8usize, raw.len(),
                    );
                    [
                        (
                            usize::try_from(raw[0usize])
                                .unwrap_or_else(|_| {
                                    panic!(
                                        "Pad: negative pad value {} at index {}", raw[0usize],
                                        0usize
                                    )
                                }),
                            usize::try_from(raw[4usize])
                                .unwrap_or_else(|_| {
                                    panic!(
                                        "Pad: negative pad value {} at index {}", raw[4usize],
                                        4usize
                                    )
                                }),
                        ),
                        (
                            usize::try_from(raw[1usize])
                                .unwrap_or_else(|_| {
                                    panic!(
                                        "Pad: negative pad value {} at index {}", raw[1usize],
                                        1usize
                                    )
                                }),
                            usize::try_from(raw[5usize])
                                .unwrap_or_else(|_| {
                                    panic!(
                                        "Pad: negative pad value {} at index {}", raw[5usize],
                                        5usize
                                    )
                                }),
                        ),
                        (
                            usize::try_from(raw[2usize])
                                .unwrap_or_else(|_| {
                                    panic!(
                                        "Pad: negative pad value {} at index {}", raw[2usize],
                                        2usize
                                    )
                                }),
                            usize::try_from(raw[6usize])
                                .unwrap_or_else(|_| {
                                    panic!(
                                        "Pad: negative pad value {} at index {}", raw[6usize],
                                        6usize
                                    )
                                }),
                        ),
                        (
                            usize::try_from(raw[3usize])
                                .unwrap_or_else(|_| {
                                    panic!(
                                        "Pad: negative pad value {} at index {}", raw[3usize],
                                        3usize
                                    )
                                }),
                            usize::try_from(raw[7usize])
                                .unwrap_or_else(|_| {
                                    panic!(
                                        "Pad: negative pad value {} at index {}", raw[7usize],
                                        7usize
                                    )
                                }),
                        ),
                    ]
                },
                burn::tensor::ops::PadMode::Constant(0f32),
            );
        let unsqueeze140_out1 = [gather34_out1 as i64];
        let unsqueeze141_out1 = [gather36_out1 as i64];
        let constant671_out1: [i64; 1] = [5i64];
        let constant672_out1: [i64; 1] = [14i64];
        let constant673_out1: [i64; 1] = [5i64];
        let constant674_out1: [i64; 1] = [14i64];
        let concat74_out1: [i64; 6usize] = [
            &unsqueeze140_out1[..],
            &constant671_out1[..],
            &constant672_out1[..],
            &constant673_out1[..],
            &constant674_out1[..],
            &unsqueeze141_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape81_out1 = pad8_out1.reshape(concat74_out1);
        let transpose61_out1 = reshape81_out1.permute([0, 1, 3, 2, 4, 5]);
        let constant676_out1 = 5i64;
        let mul53_out1 = gather34_out1 * constant676_out1;
        let constant677_out1 = 5i64;
        let mul54_out1 = mul53_out1 * constant677_out1;
        let unsqueeze142_out1 = [mul54_out1 as i64];
        let unsqueeze143_out1 = [gather36_out1 as i64];
        let constant679_out1: [i64; 1] = [196i64];
        let concat75_out1: [i64; 3usize] = [
            &unsqueeze142_out1[..],
            &constant679_out1[..],
            &unsqueeze143_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape82_out1 = transpose61_out1.reshape(concat75_out1);
        let shape39_out1: [i64; 3] = {
            let axes = &reshape82_out1.clone().dims()[0..3];
            let mut output = [0i64; 3];
            for i in 0..3 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let gather37_out1 = shape39_out1[0] as i64;
        let gather38_out1 = shape39_out1[1] as i64;
        let layernormalization13_out1 = {
            let dtype = reshape82_out1.dtype();
            self.layernormalization13
                .forward(reshape82_out1.cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let linear25_out1 = self.linear25.forward(layernormalization13_out1);
        let unsqueeze144_out1 = [gather37_out1 as i64];
        let unsqueeze145_out1 = [gather38_out1 as i64];
        let constant686_out1: [i64; 1] = [-1i64];
        let constant685_out1: [i64; 1] = [5i64];
        let concat76_out1: [i64; 4usize] = [
            &unsqueeze144_out1[..],
            &unsqueeze145_out1[..],
            &constant685_out1[..],
            &constant686_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape83_out1 = linear25_out1.reshape(concat76_out1);
        let split_tensors = reshape83_out1.split_with_sizes([32, 32, 32].into(), 3);
        let [split7_out1, split7_out2, split7_out3] = split_tensors.try_into().unwrap();
        let transpose62_out1 = split7_out1.permute([0, 2, 1, 3]);
        let transpose63_out1 = split7_out3.permute([0, 2, 1, 3]);
        let transpose64_out1 = split7_out2.permute([0, 2, 3, 1]);
        let matmul38_out1 = transpose62_out1.matmul(transpose64_out1);
        let constant687_out1 = self.constant687.val();
        let mul55_out1 = matmul38_out1
            .mul((constant687_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let constant688_out1 = self.constant688.val();
        let add38_out1 = mul55_out1.add((constant688_out1).unsqueeze_dims(&[0isize]));
        let softmax7_out1 = burn::tensor::activation::softmax(add38_out1, 3);
        let matmul39_out1 = softmax7_out1.matmul(transpose63_out1);
        let transpose65_out1 = matmul39_out1.permute([0, 2, 1, 3]);
        let unsqueeze146_out1 = [gather37_out1 as i64];
        let unsqueeze147_out1 = [gather38_out1 as i64];
        let constant691_out1: [i64; 1] = [160i64];
        let concat77_out1: [i64; 3usize] = [
            &unsqueeze146_out1[..],
            &unsqueeze147_out1[..],
            &constant691_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape84_out1 = transpose65_out1.reshape(concat77_out1);
        let linear26_out1 = self.linear26.forward(reshape84_out1);
        let unsqueeze148_out1 = [gather34_out1 as i64];
        let unsqueeze149_out1 = [gather36_out1 as i64];
        let constant693_out1: [i64; 1] = [5i64];
        let constant694_out1: [i64; 1] = [5i64];
        let constant695_out1: [i64; 1] = [14i64];
        let constant696_out1: [i64; 1] = [14i64];
        let concat78_out1: [i64; 6usize] = [
            &unsqueeze148_out1[..],
            &constant693_out1[..],
            &constant694_out1[..],
            &constant695_out1[..],
            &constant696_out1[..],
            &unsqueeze149_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape85_out1 = linear26_out1.reshape(concat78_out1);
        let transpose66_out1 = reshape85_out1.permute([0, 1, 3, 2, 4, 5]);
        let unsqueeze150_out1 = [gather34_out1 as i64];
        let unsqueeze151_out1 = [gather36_out1 as i64];
        let constant699_out1: [i64; 1] = [70i64];
        let constant700_out1: [i64; 1] = [70i64];
        let concat79_out1: [i64; 4usize] = [
            &unsqueeze150_out1[..],
            &constant699_out1[..],
            &constant700_out1[..],
            &unsqueeze151_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape86_out1 = transpose66_out1.reshape(concat79_out1);
        let slice23_out1 = reshape86_out1.slice(s![.., 0..64, .., ..]);
        let slice24_out1 = slice23_out1.slice(s![.., .., 0..64, ..]);
        let unsqueeze152_out1 = [gather34_out1 as i64];
        let unsqueeze153_out1 = [gather35_out1 as i64];
        let unsqueeze154_out1 = [gather36_out1 as i64];
        let concat80_out1: [i64; 3usize] = [
            &unsqueeze152_out1[..],
            &unsqueeze153_out1[..],
            &unsqueeze154_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape87_out1 = slice24_out1.reshape(concat80_out1);
        let add39_out1 = add37_out1.add(reshape87_out1);
        let transpose67_out1 = add39_out1.permute([0, 2, 1]);
        let unsqueeze155_out1 = [gather34_out1 as i64];
        let unsqueeze156_out1 = [gather36_out1 as i64];
        let constant716_out1: [i64; 1] = [64i64];
        let constant715_out1: [i64; 1] = [64i64];
        let concat81_out1: [i64; 4usize] = [
            &unsqueeze155_out1[..],
            &unsqueeze156_out1[..],
            &constant715_out1[..],
            &constant716_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape88_out1 = transpose67_out1.reshape(concat81_out1);
        let conv2d21_out1 = self.conv2d21.forward(reshape88_out1);
        let unsqueeze157_out1 = [gather34_out1 as i64];
        let unsqueeze158_out1 = [gather36_out1 as i64];
        let unsqueeze159_out1 = [gather35_out1 as i64];
        let concat82_out1: [i64; 3usize] = [
            &unsqueeze157_out1[..],
            &unsqueeze158_out1[..],
            &unsqueeze159_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape89_out1 = conv2d21_out1.reshape(concat82_out1);
        reshape89_out1
    }
}
#[derive(Module, Debug)]
pub struct Submodule9 {
    layernormalization14: LayerNorm,
    linear27: Linear,
    constant720: burn::module::Param<Tensor<1>>,
    constant721: burn::module::Param<Tensor<1>>,
    constant722: burn::module::Param<Tensor<1>>,
    linear28: Linear,
    layernormalization15: LayerNorm,
    linear29: Linear,
    constant755: burn::module::Param<Tensor<1>>,
    constant756: burn::module::Param<Tensor<3>>,
    linear30: Linear,
    conv2d22: Conv2d,
    #[module(skip)]
    device: Device,
}
impl Submodule9 {
    #[allow(unused_variables)]
    pub fn new(device: &Device) -> Self {
        let layernormalization14 = LayerNormConfig::new(160)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(true)
            .init(device);
        let linear27 = LinearConfig::new(160, 640).with_bias(true).init(device);
        let constant720: burn::module::Param<Tensor<1>> = burn::module::Param::uninitialized(
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
        let constant721: burn::module::Param<Tensor<1>> = burn::module::Param::uninitialized(
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
        let constant722: burn::module::Param<Tensor<1>> = burn::module::Param::uninitialized(
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
        let linear28 = LinearConfig::new(640, 160).with_bias(true).init(device);
        let layernormalization15 = LayerNormConfig::new(160)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(true)
            .init(device);
        let linear29 = LinearConfig::new(160, 480).with_bias(true).init(device);
        let constant755: burn::module::Param<Tensor<1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<
                1,
            >::from_data(
                burn::tensor::TensorData::from([0.1767766922712326f64]),
                (device, burn::tensor::DType::F32),
            ),
            device.clone(),
            false,
            [1].into(),
        );
        let constant756: burn::module::Param<Tensor<3>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<
                3,
            >::zeros([5, 196, 196], (device, burn::tensor::DType::F32)),
            device.clone(),
            false,
            [5, 196, 196].into(),
        );
        let linear30 = LinearConfig::new(160, 160).with_bias(true).init(device);
        let conv2d22 = Conv2dConfig::new([160, 160], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(160)
            .with_bias(true)
            .init(device);
        Self {
            layernormalization14,
            linear27,
            constant720,
            constant721,
            constant722,
            linear28,
            layernormalization15,
            linear29,
            constant755,
            constant756,
            linear30,
            conv2d22,
            device: device.clone(),
        }
    }
    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(&self, reshape89_out1: Tensor<3>) -> Tensor<3> {
        let transpose68_out1 = reshape89_out1.permute([0, 2, 1]);
        let layernormalization14_out1 = {
            let dtype = transpose68_out1.clone().dtype();
            self.layernormalization14
                .forward(transpose68_out1.clone().cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let linear27_out1 = self.linear27.forward(layernormalization14_out1);
        let constant720_out1 = self.constant720.val();
        let div19_out1 = linear27_out1
            .clone()
            .div((constant720_out1).unsqueeze_dims(&[0isize, 1isize]));
        let erf18_out1 = div19_out1.erf();
        let constant721_out1 = self.constant721.val();
        let add40_out1 = erf18_out1
            .add((constant721_out1).unsqueeze_dims(&[0isize, 1isize]));
        let mul56_out1 = linear27_out1.mul(add40_out1);
        let constant722_out1 = self.constant722.val();
        let mul57_out1 = mul56_out1
            .mul((constant722_out1).unsqueeze_dims(&[0isize, 1isize]));
        let linear28_out1 = self.linear28.forward(mul57_out1);
        let add41_out1 = transpose68_out1.add(linear28_out1);
        let shape41_out1: [i64; 3] = {
            let axes = &add41_out1.clone().dims()[0..3];
            let mut output = [0i64; 3];
            for i in 0..3 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let gather39_out1 = shape41_out1[0] as i64;
        let gather40_out1 = shape41_out1[1] as i64;
        let gather41_out1 = shape41_out1[2] as i64;
        let unsqueeze160_out1 = [gather39_out1 as i64];
        let unsqueeze161_out1 = [gather41_out1 as i64];
        let constant727_out1: [i64; 1] = [64i64];
        let constant728_out1: [i64; 1] = [64i64];
        let concat83_out1: [i64; 4usize] = [
            &unsqueeze160_out1[..],
            &constant727_out1[..],
            &constant728_out1[..],
            &unsqueeze161_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape90_out1 = add41_out1.clone().reshape(concat83_out1);
        let constantofshape9_out1: [i64; 2usize] = [0i64, 0i64];
        let constant731_out1: [i64; 6] = [0i64, 0i64, 0i64, 6i64, 0i64, 6i64];
        let concat84_out1: [i64; 8usize] = [
            &constant731_out1[..],
            &constantofshape9_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape91_out1 = {
            let shape_array = concat84_out1 as [i64; 8usize];
            Tensor::<
                1,
                Int,
            >::from_data(
                TensorData::from(shape_array),
                (&self.device, burn::tensor::DType::I64),
            )
        }
            .reshape([-1, 2]);
        let slice25_out1 = {
            let slice_input = reshape91_out1;
            let slice_dims = slice_input.dims();
            let reverse_bounds = |
                dim: usize,
                start: i64,
                end: i64,
            | -> core::ops::Range<usize> {
                if dim == 0 {
                    return 0..0;
                }
                let dim = dim as i64;
                let start = if start < 0 { start.saturating_add(dim) } else { start };
                let end = if end < 0 { end.saturating_add(dim) } else { end };
                let hi = start.clamp(-1, dim - 1) + 1;
                let lo = (end.clamp(-1, dim - 1) + 1).min(hi);
                (lo as usize)..(hi as usize)
            };
            slice_input
                .slice(
                    s![
                        reverse_bounds(slice_dims[0], - 1i64, - 9223372036854775807i64);
                        - 1, ..
                    ],
                )
        };
        let transpose69_out1 = slice25_out1.permute([1, 0]);
        let reshape92_out1 = transpose69_out1.reshape([-1]);
        let pad9_out1 = reshape90_out1
            .pad(
                {
                    let raw: alloc::vec::Vec<i64> = reshape92_out1
                        .to_data()
                        .convert::<i64>()
                        .try_into_vec::<i64>()
                        .unwrap();
                    assert_eq!(
                        raw.len(), 8usize,
                        "Pad: runtime pads length mismatch (expected {}, got {})",
                        8usize, raw.len(),
                    );
                    [
                        (
                            usize::try_from(raw[0usize])
                                .unwrap_or_else(|_| {
                                    panic!(
                                        "Pad: negative pad value {} at index {}", raw[0usize],
                                        0usize
                                    )
                                }),
                            usize::try_from(raw[4usize])
                                .unwrap_or_else(|_| {
                                    panic!(
                                        "Pad: negative pad value {} at index {}", raw[4usize],
                                        4usize
                                    )
                                }),
                        ),
                        (
                            usize::try_from(raw[1usize])
                                .unwrap_or_else(|_| {
                                    panic!(
                                        "Pad: negative pad value {} at index {}", raw[1usize],
                                        1usize
                                    )
                                }),
                            usize::try_from(raw[5usize])
                                .unwrap_or_else(|_| {
                                    panic!(
                                        "Pad: negative pad value {} at index {}", raw[5usize],
                                        5usize
                                    )
                                }),
                        ),
                        (
                            usize::try_from(raw[2usize])
                                .unwrap_or_else(|_| {
                                    panic!(
                                        "Pad: negative pad value {} at index {}", raw[2usize],
                                        2usize
                                    )
                                }),
                            usize::try_from(raw[6usize])
                                .unwrap_or_else(|_| {
                                    panic!(
                                        "Pad: negative pad value {} at index {}", raw[6usize],
                                        6usize
                                    )
                                }),
                        ),
                        (
                            usize::try_from(raw[3usize])
                                .unwrap_or_else(|_| {
                                    panic!(
                                        "Pad: negative pad value {} at index {}", raw[3usize],
                                        3usize
                                    )
                                }),
                            usize::try_from(raw[7usize])
                                .unwrap_or_else(|_| {
                                    panic!(
                                        "Pad: negative pad value {} at index {}", raw[7usize],
                                        7usize
                                    )
                                }),
                        ),
                    ]
                },
                burn::tensor::ops::PadMode::Constant(0f32),
            );
        let unsqueeze162_out1 = [gather39_out1 as i64];
        let unsqueeze163_out1 = [gather41_out1 as i64];
        let constant739_out1: [i64; 1] = [5i64];
        let constant740_out1: [i64; 1] = [14i64];
        let constant741_out1: [i64; 1] = [5i64];
        let constant742_out1: [i64; 1] = [14i64];
        let concat85_out1: [i64; 6usize] = [
            &unsqueeze162_out1[..],
            &constant739_out1[..],
            &constant740_out1[..],
            &constant741_out1[..],
            &constant742_out1[..],
            &unsqueeze163_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape93_out1 = pad9_out1.reshape(concat85_out1);
        let transpose70_out1 = reshape93_out1.permute([0, 1, 3, 2, 4, 5]);
        let constant744_out1 = 5i64;
        let mul58_out1 = gather39_out1 * constant744_out1;
        let constant745_out1 = 5i64;
        let mul59_out1 = mul58_out1 * constant745_out1;
        let unsqueeze164_out1 = [mul59_out1 as i64];
        let unsqueeze165_out1 = [gather41_out1 as i64];
        let constant747_out1: [i64; 1] = [196i64];
        let concat86_out1: [i64; 3usize] = [
            &unsqueeze164_out1[..],
            &constant747_out1[..],
            &unsqueeze165_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape94_out1 = transpose70_out1.reshape(concat86_out1);
        let shape44_out1: [i64; 3] = {
            let axes = &reshape94_out1.clone().dims()[0..3];
            let mut output = [0i64; 3];
            for i in 0..3 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let gather42_out1 = shape44_out1[0] as i64;
        let gather43_out1 = shape44_out1[1] as i64;
        let layernormalization15_out1 = {
            let dtype = reshape94_out1.dtype();
            self.layernormalization15
                .forward(reshape94_out1.cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let linear29_out1 = self.linear29.forward(layernormalization15_out1);
        let unsqueeze166_out1 = [gather42_out1 as i64];
        let unsqueeze167_out1 = [gather43_out1 as i64];
        let constant754_out1: [i64; 1] = [-1i64];
        let constant753_out1: [i64; 1] = [5i64];
        let concat87_out1: [i64; 4usize] = [
            &unsqueeze166_out1[..],
            &unsqueeze167_out1[..],
            &constant753_out1[..],
            &constant754_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape95_out1 = linear29_out1.reshape(concat87_out1);
        let split_tensors = reshape95_out1.split_with_sizes([32, 32, 32].into(), 3);
        let [split8_out1, split8_out2, split8_out3] = split_tensors.try_into().unwrap();
        let transpose71_out1 = split8_out1.permute([0, 2, 1, 3]);
        let transpose72_out1 = split8_out3.permute([0, 2, 1, 3]);
        let transpose73_out1 = split8_out2.permute([0, 2, 3, 1]);
        let matmul44_out1 = transpose71_out1.matmul(transpose73_out1);
        let constant755_out1 = self.constant755.val();
        let mul60_out1 = matmul44_out1
            .mul((constant755_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let constant756_out1 = self.constant756.val();
        let add42_out1 = mul60_out1.add((constant756_out1).unsqueeze_dims(&[0isize]));
        let softmax8_out1 = burn::tensor::activation::softmax(add42_out1, 3);
        let matmul45_out1 = softmax8_out1.matmul(transpose72_out1);
        let transpose74_out1 = matmul45_out1.permute([0, 2, 1, 3]);
        let unsqueeze168_out1 = [gather42_out1 as i64];
        let unsqueeze169_out1 = [gather43_out1 as i64];
        let constant759_out1: [i64; 1] = [160i64];
        let concat88_out1: [i64; 3usize] = [
            &unsqueeze168_out1[..],
            &unsqueeze169_out1[..],
            &constant759_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape96_out1 = transpose74_out1.reshape(concat88_out1);
        let linear30_out1 = self.linear30.forward(reshape96_out1);
        let unsqueeze170_out1 = [gather39_out1 as i64];
        let unsqueeze171_out1 = [gather41_out1 as i64];
        let constant761_out1: [i64; 1] = [5i64];
        let constant762_out1: [i64; 1] = [5i64];
        let constant763_out1: [i64; 1] = [14i64];
        let constant764_out1: [i64; 1] = [14i64];
        let concat89_out1: [i64; 6usize] = [
            &unsqueeze170_out1[..],
            &constant761_out1[..],
            &constant762_out1[..],
            &constant763_out1[..],
            &constant764_out1[..],
            &unsqueeze171_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape97_out1 = linear30_out1.reshape(concat89_out1);
        let transpose75_out1 = reshape97_out1.permute([0, 1, 3, 2, 4, 5]);
        let unsqueeze172_out1 = [gather39_out1 as i64];
        let unsqueeze173_out1 = [gather41_out1 as i64];
        let constant767_out1: [i64; 1] = [70i64];
        let constant768_out1: [i64; 1] = [70i64];
        let concat90_out1: [i64; 4usize] = [
            &unsqueeze172_out1[..],
            &constant767_out1[..],
            &constant768_out1[..],
            &unsqueeze173_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape98_out1 = transpose75_out1.reshape(concat90_out1);
        let slice26_out1 = reshape98_out1.slice(s![.., 0..64, .., ..]);
        let slice27_out1 = slice26_out1.slice(s![.., .., 0..64, ..]);
        let unsqueeze174_out1 = [gather39_out1 as i64];
        let unsqueeze175_out1 = [gather40_out1 as i64];
        let unsqueeze176_out1 = [gather41_out1 as i64];
        let concat91_out1: [i64; 3usize] = [
            &unsqueeze174_out1[..],
            &unsqueeze175_out1[..],
            &unsqueeze176_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape99_out1 = slice27_out1.reshape(concat91_out1);
        let add43_out1 = add41_out1.add(reshape99_out1);
        let transpose76_out1 = add43_out1.permute([0, 2, 1]);
        let unsqueeze177_out1 = [gather39_out1 as i64];
        let unsqueeze178_out1 = [gather41_out1 as i64];
        let constant784_out1: [i64; 1] = [64i64];
        let constant783_out1: [i64; 1] = [64i64];
        let concat92_out1: [i64; 4usize] = [
            &unsqueeze177_out1[..],
            &unsqueeze178_out1[..],
            &constant783_out1[..],
            &constant784_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape100_out1 = transpose76_out1.reshape(concat92_out1);
        let conv2d22_out1 = self.conv2d22.forward(reshape100_out1);
        let unsqueeze179_out1 = [gather39_out1 as i64];
        let unsqueeze180_out1 = [gather41_out1 as i64];
        let unsqueeze181_out1 = [gather40_out1 as i64];
        let concat93_out1: [i64; 3usize] = [
            &unsqueeze179_out1[..],
            &unsqueeze180_out1[..],
            &unsqueeze181_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape101_out1 = conv2d22_out1.reshape(concat93_out1);
        reshape101_out1
    }
}
#[derive(Module, Debug)]
pub struct Submodule10 {
    layernormalization16: LayerNorm,
    linear31: Linear,
    constant788: burn::module::Param<Tensor<1>>,
    constant789: burn::module::Param<Tensor<1>>,
    constant790: burn::module::Param<Tensor<1>>,
    linear32: Linear,
    conv2d23: Conv2d,
    constant792: burn::module::Param<Tensor<1>>,
    constant793: burn::module::Param<Tensor<1>>,
    constant794: burn::module::Param<Tensor<1>>,
    conv2d24: Conv2d,
    constant795: burn::module::Param<Tensor<1>>,
    constant796: burn::module::Param<Tensor<1>>,
    constant797: burn::module::Param<Tensor<1>>,
    conv2d25: Conv2d,
    layernormalization17: LayerNorm,
    linear33: Linear,
    constant834: burn::module::Param<Tensor<1>>,
    constant835: burn::module::Param<Tensor<3>>,
    linear34: Linear,
    conv2d26: Conv2d,
    #[module(skip)]
    device: Device,
}
impl Submodule10 {
    #[allow(unused_variables)]
    pub fn new(device: &Device) -> Self {
        let layernormalization16 = LayerNormConfig::new(160)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(true)
            .init(device);
        let linear31 = LinearConfig::new(160, 640).with_bias(true).init(device);
        let constant788: burn::module::Param<Tensor<1>> = burn::module::Param::uninitialized(
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
        let constant789: burn::module::Param<Tensor<1>> = burn::module::Param::uninitialized(
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
        let constant790: burn::module::Param<Tensor<1>> = burn::module::Param::uninitialized(
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
        let linear32 = LinearConfig::new(640, 160).with_bias(true).init(device);
        let conv2d23 = Conv2dConfig::new([160, 320], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let constant792: burn::module::Param<Tensor<1>> = burn::module::Param::uninitialized(
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
        let constant793: burn::module::Param<Tensor<1>> = burn::module::Param::uninitialized(
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
        let constant794: burn::module::Param<Tensor<1>> = burn::module::Param::uninitialized(
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
        let conv2d24 = Conv2dConfig::new([320, 320], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(320)
            .with_bias(true)
            .init(device);
        let constant795: burn::module::Param<Tensor<1>> = burn::module::Param::uninitialized(
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
        let constant796: burn::module::Param<Tensor<1>> = burn::module::Param::uninitialized(
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
        let constant797: burn::module::Param<Tensor<1>> = burn::module::Param::uninitialized(
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
        let conv2d25 = Conv2dConfig::new([320, 320], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let layernormalization17 = LayerNormConfig::new(320)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(true)
            .init(device);
        let linear33 = LinearConfig::new(320, 960).with_bias(true).init(device);
        let constant834: burn::module::Param<Tensor<1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<
                1,
            >::from_data(
                burn::tensor::TensorData::from([0.1767766922712326f64]),
                (device, burn::tensor::DType::F32),
            ),
            device.clone(),
            false,
            [1].into(),
        );
        let constant835: burn::module::Param<Tensor<3>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<
                3,
            >::zeros([10, 49, 49], (device, burn::tensor::DType::F32)),
            device.clone(),
            false,
            [10, 49, 49].into(),
        );
        let linear34 = LinearConfig::new(320, 320).with_bias(true).init(device);
        let conv2d26 = Conv2dConfig::new([320, 320], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(320)
            .with_bias(true)
            .init(device);
        Self {
            layernormalization16,
            linear31,
            constant788,
            constant789,
            constant790,
            linear32,
            conv2d23,
            constant792,
            constant793,
            constant794,
            conv2d24,
            constant795,
            constant796,
            constant797,
            conv2d25,
            layernormalization17,
            linear33,
            constant834,
            constant835,
            linear34,
            conv2d26,
            device: device.clone(),
        }
    }
    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(&self, reshape101_out1: Tensor<3>) -> Tensor<3> {
        let transpose77_out1 = reshape101_out1.permute([0, 2, 1]);
        let layernormalization16_out1 = {
            let dtype = transpose77_out1.clone().dtype();
            self.layernormalization16
                .forward(transpose77_out1.clone().cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let linear31_out1 = self.linear31.forward(layernormalization16_out1);
        let constant788_out1 = self.constant788.val();
        let div20_out1 = linear31_out1
            .clone()
            .div((constant788_out1).unsqueeze_dims(&[0isize, 1isize]));
        let erf19_out1 = div20_out1.erf();
        let constant789_out1 = self.constant789.val();
        let add44_out1 = erf19_out1
            .add((constant789_out1).unsqueeze_dims(&[0isize, 1isize]));
        let mul61_out1 = linear31_out1.mul(add44_out1);
        let constant790_out1 = self.constant790.val();
        let mul62_out1 = mul61_out1
            .mul((constant790_out1).unsqueeze_dims(&[0isize, 1isize]));
        let linear32_out1 = self.linear32.forward(mul62_out1);
        let add45_out1 = transpose77_out1.add(linear32_out1);
        let reshape102_out1 = add45_out1.reshape([1, 64, 64, -1]);
        let transpose78_out1 = reshape102_out1.permute([0, 3, 1, 2]);
        let conv2d23_out1 = self.conv2d23.forward(transpose78_out1);
        let constant792_out1 = self.constant792.val();
        let div21_out1 = conv2d23_out1
            .clone()
            .div((constant792_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let erf20_out1 = div21_out1.erf();
        let constant793_out1 = self.constant793.val();
        let add46_out1 = erf20_out1
            .add((constant793_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let mul63_out1 = conv2d23_out1.mul(add46_out1);
        let constant794_out1 = self.constant794.val();
        let mul64_out1 = mul63_out1
            .mul((constant794_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let conv2d24_out1 = self.conv2d24.forward(mul64_out1);
        let constant795_out1 = self.constant795.val();
        let div22_out1 = conv2d24_out1
            .clone()
            .div((constant795_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let erf21_out1 = div22_out1.erf();
        let constant796_out1 = self.constant796.val();
        let add47_out1 = erf21_out1
            .add((constant796_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let mul65_out1 = conv2d24_out1.mul(add47_out1);
        let constant797_out1 = self.constant797.val();
        let mul66_out1 = mul65_out1
            .mul((constant797_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let conv2d25_out1 = self.conv2d25.forward(mul66_out1);
        let reshape103_out1 = conv2d25_out1.reshape([1, 320, -1]);
        let transpose79_out1 = reshape103_out1.permute([0, 2, 1]);
        let shape47_out1: [i64; 3] = {
            let axes = &transpose79_out1.clone().dims()[0..3];
            let mut output = [0i64; 3];
            for i in 0..3 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let gather44_out1 = shape47_out1[0] as i64;
        let gather45_out1 = shape47_out1[1] as i64;
        let gather46_out1 = shape47_out1[2] as i64;
        let unsqueeze182_out1 = [gather44_out1 as i64];
        let unsqueeze183_out1 = [gather46_out1 as i64];
        let constant806_out1: [i64; 1] = [64i64];
        let constant807_out1: [i64; 1] = [64i64];
        let concat95_out1: [i64; 4usize] = [
            &unsqueeze182_out1[..],
            &constant806_out1[..],
            &constant807_out1[..],
            &unsqueeze183_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape104_out1 = transpose79_out1.clone().reshape(concat95_out1);
        let constantofshape10_out1: [i64; 2usize] = [0i64, 0i64];
        let constant810_out1: [i64; 6] = [0i64, 0i64, 0i64, 6i64, 0i64, 6i64];
        let concat96_out1: [i64; 8usize] = [
            &constant810_out1[..],
            &constantofshape10_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape105_out1 = {
            let shape_array = concat96_out1 as [i64; 8usize];
            Tensor::<
                1,
                Int,
            >::from_data(
                TensorData::from(shape_array),
                (&self.device, burn::tensor::DType::I64),
            )
        }
            .reshape([-1, 2]);
        let slice29_out1 = {
            let slice_input = reshape105_out1;
            let slice_dims = slice_input.dims();
            let reverse_bounds = |
                dim: usize,
                start: i64,
                end: i64,
            | -> core::ops::Range<usize> {
                if dim == 0 {
                    return 0..0;
                }
                let dim = dim as i64;
                let start = if start < 0 { start.saturating_add(dim) } else { start };
                let end = if end < 0 { end.saturating_add(dim) } else { end };
                let hi = start.clamp(-1, dim - 1) + 1;
                let lo = (end.clamp(-1, dim - 1) + 1).min(hi);
                (lo as usize)..(hi as usize)
            };
            slice_input
                .slice(
                    s![
                        reverse_bounds(slice_dims[0], - 1i64, - 9223372036854775807i64);
                        - 1, ..
                    ],
                )
        };
        let transpose80_out1 = slice29_out1.permute([1, 0]);
        let reshape106_out1 = transpose80_out1.reshape([-1]);
        let pad10_out1 = reshape104_out1
            .pad(
                {
                    let raw: alloc::vec::Vec<i64> = reshape106_out1
                        .to_data()
                        .convert::<i64>()
                        .try_into_vec::<i64>()
                        .unwrap();
                    assert_eq!(
                        raw.len(), 8usize,
                        "Pad: runtime pads length mismatch (expected {}, got {})",
                        8usize, raw.len(),
                    );
                    [
                        (
                            usize::try_from(raw[0usize])
                                .unwrap_or_else(|_| {
                                    panic!(
                                        "Pad: negative pad value {} at index {}", raw[0usize],
                                        0usize
                                    )
                                }),
                            usize::try_from(raw[4usize])
                                .unwrap_or_else(|_| {
                                    panic!(
                                        "Pad: negative pad value {} at index {}", raw[4usize],
                                        4usize
                                    )
                                }),
                        ),
                        (
                            usize::try_from(raw[1usize])
                                .unwrap_or_else(|_| {
                                    panic!(
                                        "Pad: negative pad value {} at index {}", raw[1usize],
                                        1usize
                                    )
                                }),
                            usize::try_from(raw[5usize])
                                .unwrap_or_else(|_| {
                                    panic!(
                                        "Pad: negative pad value {} at index {}", raw[5usize],
                                        5usize
                                    )
                                }),
                        ),
                        (
                            usize::try_from(raw[2usize])
                                .unwrap_or_else(|_| {
                                    panic!(
                                        "Pad: negative pad value {} at index {}", raw[2usize],
                                        2usize
                                    )
                                }),
                            usize::try_from(raw[6usize])
                                .unwrap_or_else(|_| {
                                    panic!(
                                        "Pad: negative pad value {} at index {}", raw[6usize],
                                        6usize
                                    )
                                }),
                        ),
                        (
                            usize::try_from(raw[3usize])
                                .unwrap_or_else(|_| {
                                    panic!(
                                        "Pad: negative pad value {} at index {}", raw[3usize],
                                        3usize
                                    )
                                }),
                            usize::try_from(raw[7usize])
                                .unwrap_or_else(|_| {
                                    panic!(
                                        "Pad: negative pad value {} at index {}", raw[7usize],
                                        7usize
                                    )
                                }),
                        ),
                    ]
                },
                burn::tensor::ops::PadMode::Constant(0f32),
            );
        let unsqueeze184_out1 = [gather44_out1 as i64];
        let unsqueeze185_out1 = [gather46_out1 as i64];
        let constant818_out1: [i64; 1] = [10i64];
        let constant819_out1: [i64; 1] = [7i64];
        let constant820_out1: [i64; 1] = [10i64];
        let constant821_out1: [i64; 1] = [7i64];
        let concat97_out1: [i64; 6usize] = [
            &unsqueeze184_out1[..],
            &constant818_out1[..],
            &constant819_out1[..],
            &constant820_out1[..],
            &constant821_out1[..],
            &unsqueeze185_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape107_out1 = pad10_out1.reshape(concat97_out1);
        let transpose81_out1 = reshape107_out1.permute([0, 1, 3, 2, 4, 5]);
        let constant823_out1 = 10i64;
        let mul67_out1 = gather44_out1 * constant823_out1;
        let constant824_out1 = 10i64;
        let mul68_out1 = mul67_out1 * constant824_out1;
        let unsqueeze186_out1 = [mul68_out1 as i64];
        let unsqueeze187_out1 = [gather46_out1 as i64];
        let constant826_out1: [i64; 1] = [49i64];
        let concat98_out1: [i64; 3usize] = [
            &unsqueeze186_out1[..],
            &constant826_out1[..],
            &unsqueeze187_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape108_out1 = transpose81_out1.reshape(concat98_out1);
        let shape50_out1: [i64; 3] = {
            let axes = &reshape108_out1.clone().dims()[0..3];
            let mut output = [0i64; 3];
            for i in 0..3 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let gather47_out1 = shape50_out1[0] as i64;
        let gather48_out1 = shape50_out1[1] as i64;
        let layernormalization17_out1 = {
            let dtype = reshape108_out1.dtype();
            self.layernormalization17
                .forward(reshape108_out1.cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let linear33_out1 = self.linear33.forward(layernormalization17_out1);
        let unsqueeze188_out1 = [gather47_out1 as i64];
        let unsqueeze189_out1 = [gather48_out1 as i64];
        let constant833_out1: [i64; 1] = [-1i64];
        let constant832_out1: [i64; 1] = [10i64];
        let concat99_out1: [i64; 4usize] = [
            &unsqueeze188_out1[..],
            &unsqueeze189_out1[..],
            &constant832_out1[..],
            &constant833_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape109_out1 = linear33_out1.reshape(concat99_out1);
        let split_tensors = reshape109_out1.split_with_sizes([32, 32, 32].into(), 3);
        let [split9_out1, split9_out2, split9_out3] = split_tensors.try_into().unwrap();
        let transpose82_out1 = split9_out1.permute([0, 2, 1, 3]);
        let transpose83_out1 = split9_out3.permute([0, 2, 1, 3]);
        let transpose84_out1 = split9_out2.permute([0, 2, 3, 1]);
        let matmul50_out1 = transpose82_out1.matmul(transpose84_out1);
        let constant834_out1 = self.constant834.val();
        let mul69_out1 = matmul50_out1
            .mul((constant834_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let constant835_out1 = self.constant835.val();
        let add48_out1 = mul69_out1.add((constant835_out1).unsqueeze_dims(&[0isize]));
        let softmax9_out1 = burn::tensor::activation::softmax(add48_out1, 3);
        let matmul51_out1 = softmax9_out1.matmul(transpose83_out1);
        let transpose85_out1 = matmul51_out1.permute([0, 2, 1, 3]);
        let unsqueeze190_out1 = [gather47_out1 as i64];
        let unsqueeze191_out1 = [gather48_out1 as i64];
        let constant838_out1: [i64; 1] = [320i64];
        let concat100_out1: [i64; 3usize] = [
            &unsqueeze190_out1[..],
            &unsqueeze191_out1[..],
            &constant838_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape110_out1 = transpose85_out1.reshape(concat100_out1);
        let linear34_out1 = self.linear34.forward(reshape110_out1);
        let unsqueeze192_out1 = [gather44_out1 as i64];
        let unsqueeze193_out1 = [gather46_out1 as i64];
        let constant840_out1: [i64; 1] = [10i64];
        let constant841_out1: [i64; 1] = [10i64];
        let constant842_out1: [i64; 1] = [7i64];
        let constant843_out1: [i64; 1] = [7i64];
        let concat101_out1: [i64; 6usize] = [
            &unsqueeze192_out1[..],
            &constant840_out1[..],
            &constant841_out1[..],
            &constant842_out1[..],
            &constant843_out1[..],
            &unsqueeze193_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape111_out1 = linear34_out1.reshape(concat101_out1);
        let transpose86_out1 = reshape111_out1.permute([0, 1, 3, 2, 4, 5]);
        let unsqueeze194_out1 = [gather44_out1 as i64];
        let unsqueeze195_out1 = [gather46_out1 as i64];
        let constant846_out1: [i64; 1] = [70i64];
        let constant847_out1: [i64; 1] = [70i64];
        let concat102_out1: [i64; 4usize] = [
            &unsqueeze194_out1[..],
            &constant846_out1[..],
            &constant847_out1[..],
            &unsqueeze195_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape112_out1 = transpose86_out1.reshape(concat102_out1);
        let slice30_out1 = reshape112_out1.slice(s![.., 0..64, .., ..]);
        let slice31_out1 = slice30_out1.slice(s![.., .., 0..64, ..]);
        let unsqueeze196_out1 = [gather44_out1 as i64];
        let unsqueeze197_out1 = [gather45_out1 as i64];
        let unsqueeze198_out1 = [gather46_out1 as i64];
        let concat103_out1: [i64; 3usize] = [
            &unsqueeze196_out1[..],
            &unsqueeze197_out1[..],
            &unsqueeze198_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape113_out1 = slice31_out1.reshape(concat103_out1);
        let add49_out1 = transpose79_out1.add(reshape113_out1);
        let transpose87_out1 = add49_out1.permute([0, 2, 1]);
        let unsqueeze199_out1 = [gather44_out1 as i64];
        let unsqueeze200_out1 = [gather46_out1 as i64];
        let constant863_out1: [i64; 1] = [64i64];
        let constant862_out1: [i64; 1] = [64i64];
        let concat104_out1: [i64; 4usize] = [
            &unsqueeze199_out1[..],
            &unsqueeze200_out1[..],
            &constant862_out1[..],
            &constant863_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape114_out1 = transpose87_out1.reshape(concat104_out1);
        let conv2d26_out1 = self.conv2d26.forward(reshape114_out1);
        let unsqueeze201_out1 = [gather44_out1 as i64];
        let unsqueeze202_out1 = [gather46_out1 as i64];
        let unsqueeze203_out1 = [gather45_out1 as i64];
        let concat105_out1: [i64; 3usize] = [
            &unsqueeze201_out1[..],
            &unsqueeze202_out1[..],
            &unsqueeze203_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape115_out1 = conv2d26_out1.reshape(concat105_out1);
        reshape115_out1
    }
}
#[derive(Module, Debug)]
pub struct Submodule11 {
    layernormalization18: LayerNorm,
    linear35: Linear,
    constant867: burn::module::Param<Tensor<1>>,
    constant868: burn::module::Param<Tensor<1>>,
    constant869: burn::module::Param<Tensor<1>>,
    linear36: Linear,
    layernormalization19: LayerNorm,
    linear37: Linear,
    constant902: burn::module::Param<Tensor<1>>,
    constant903: burn::module::Param<Tensor<3>>,
    linear38: Linear,
    #[module(skip)]
    device: Device,
}
impl Submodule11 {
    #[allow(unused_variables)]
    pub fn new(device: &Device) -> Self {
        let layernormalization18 = LayerNormConfig::new(320)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(true)
            .init(device);
        let linear35 = LinearConfig::new(320, 1280).with_bias(true).init(device);
        let constant867: burn::module::Param<Tensor<1>> = burn::module::Param::uninitialized(
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
        let constant868: burn::module::Param<Tensor<1>> = burn::module::Param::uninitialized(
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
        let constant869: burn::module::Param<Tensor<1>> = burn::module::Param::uninitialized(
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
        let linear36 = LinearConfig::new(1280, 320).with_bias(true).init(device);
        let layernormalization19 = LayerNormConfig::new(320)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(true)
            .init(device);
        let linear37 = LinearConfig::new(320, 960).with_bias(true).init(device);
        let constant902: burn::module::Param<Tensor<1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<
                1,
            >::from_data(
                burn::tensor::TensorData::from([0.1767766922712326f64]),
                (device, burn::tensor::DType::F32),
            ),
            device.clone(),
            false,
            [1].into(),
        );
        let constant903: burn::module::Param<Tensor<3>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<
                3,
            >::zeros([10, 49, 49], (device, burn::tensor::DType::F32)),
            device.clone(),
            false,
            [10, 49, 49].into(),
        );
        let linear38 = LinearConfig::new(320, 320).with_bias(true).init(device);
        Self {
            layernormalization18,
            linear35,
            constant867,
            constant868,
            constant869,
            linear36,
            layernormalization19,
            linear37,
            constant902,
            constant903,
            linear38,
            device: device.clone(),
        }
    }
    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(&self, reshape115_out1: Tensor<3>) -> (Tensor<3>, i64, i64, i64) {
        let transpose88_out1 = reshape115_out1.permute([0, 2, 1]);
        let layernormalization18_out1 = {
            let dtype = transpose88_out1.clone().dtype();
            self.layernormalization18
                .forward(transpose88_out1.clone().cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let linear35_out1 = self.linear35.forward(layernormalization18_out1);
        let constant867_out1 = self.constant867.val();
        let div23_out1 = linear35_out1
            .clone()
            .div((constant867_out1).unsqueeze_dims(&[0isize, 1isize]));
        let erf22_out1 = div23_out1.erf();
        let constant868_out1 = self.constant868.val();
        let add50_out1 = erf22_out1
            .add((constant868_out1).unsqueeze_dims(&[0isize, 1isize]));
        let mul70_out1 = linear35_out1.mul(add50_out1);
        let constant869_out1 = self.constant869.val();
        let mul71_out1 = mul70_out1
            .mul((constant869_out1).unsqueeze_dims(&[0isize, 1isize]));
        let linear36_out1 = self.linear36.forward(mul71_out1);
        let add51_out1 = transpose88_out1.add(linear36_out1);
        let shape52_out1: [i64; 3] = {
            let axes = &add51_out1.clone().dims()[0..3];
            let mut output = [0i64; 3];
            for i in 0..3 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let gather49_out1 = shape52_out1[0] as i64;
        let gather50_out1 = shape52_out1[1] as i64;
        let gather51_out1 = shape52_out1[2] as i64;
        let unsqueeze204_out1 = [gather49_out1 as i64];
        let unsqueeze205_out1 = [gather51_out1 as i64];
        let constant874_out1: [i64; 1] = [64i64];
        let constant875_out1: [i64; 1] = [64i64];
        let concat106_out1: [i64; 4usize] = [
            &unsqueeze204_out1[..],
            &constant874_out1[..],
            &constant875_out1[..],
            &unsqueeze205_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape116_out1 = add51_out1.clone().reshape(concat106_out1);
        let constantofshape11_out1: [i64; 2usize] = [0i64, 0i64];
        let constant878_out1: [i64; 6] = [0i64, 0i64, 0i64, 6i64, 0i64, 6i64];
        let concat107_out1: [i64; 8usize] = [
            &constant878_out1[..],
            &constantofshape11_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape117_out1 = {
            let shape_array = concat107_out1 as [i64; 8usize];
            Tensor::<
                1,
                Int,
            >::from_data(
                TensorData::from(shape_array),
                (&self.device, burn::tensor::DType::I64),
            )
        }
            .reshape([-1, 2]);
        let slice32_out1 = {
            let slice_input = reshape117_out1;
            let slice_dims = slice_input.dims();
            let reverse_bounds = |
                dim: usize,
                start: i64,
                end: i64,
            | -> core::ops::Range<usize> {
                if dim == 0 {
                    return 0..0;
                }
                let dim = dim as i64;
                let start = if start < 0 { start.saturating_add(dim) } else { start };
                let end = if end < 0 { end.saturating_add(dim) } else { end };
                let hi = start.clamp(-1, dim - 1) + 1;
                let lo = (end.clamp(-1, dim - 1) + 1).min(hi);
                (lo as usize)..(hi as usize)
            };
            slice_input
                .slice(
                    s![
                        reverse_bounds(slice_dims[0], - 1i64, - 9223372036854775807i64);
                        - 1, ..
                    ],
                )
        };
        let transpose89_out1 = slice32_out1.permute([1, 0]);
        let reshape118_out1 = transpose89_out1.reshape([-1]);
        let pad11_out1 = reshape116_out1
            .pad(
                {
                    let raw: alloc::vec::Vec<i64> = reshape118_out1
                        .to_data()
                        .convert::<i64>()
                        .try_into_vec::<i64>()
                        .unwrap();
                    assert_eq!(
                        raw.len(), 8usize,
                        "Pad: runtime pads length mismatch (expected {}, got {})",
                        8usize, raw.len(),
                    );
                    [
                        (
                            usize::try_from(raw[0usize])
                                .unwrap_or_else(|_| {
                                    panic!(
                                        "Pad: negative pad value {} at index {}", raw[0usize],
                                        0usize
                                    )
                                }),
                            usize::try_from(raw[4usize])
                                .unwrap_or_else(|_| {
                                    panic!(
                                        "Pad: negative pad value {} at index {}", raw[4usize],
                                        4usize
                                    )
                                }),
                        ),
                        (
                            usize::try_from(raw[1usize])
                                .unwrap_or_else(|_| {
                                    panic!(
                                        "Pad: negative pad value {} at index {}", raw[1usize],
                                        1usize
                                    )
                                }),
                            usize::try_from(raw[5usize])
                                .unwrap_or_else(|_| {
                                    panic!(
                                        "Pad: negative pad value {} at index {}", raw[5usize],
                                        5usize
                                    )
                                }),
                        ),
                        (
                            usize::try_from(raw[2usize])
                                .unwrap_or_else(|_| {
                                    panic!(
                                        "Pad: negative pad value {} at index {}", raw[2usize],
                                        2usize
                                    )
                                }),
                            usize::try_from(raw[6usize])
                                .unwrap_or_else(|_| {
                                    panic!(
                                        "Pad: negative pad value {} at index {}", raw[6usize],
                                        6usize
                                    )
                                }),
                        ),
                        (
                            usize::try_from(raw[3usize])
                                .unwrap_or_else(|_| {
                                    panic!(
                                        "Pad: negative pad value {} at index {}", raw[3usize],
                                        3usize
                                    )
                                }),
                            usize::try_from(raw[7usize])
                                .unwrap_or_else(|_| {
                                    panic!(
                                        "Pad: negative pad value {} at index {}", raw[7usize],
                                        7usize
                                    )
                                }),
                        ),
                    ]
                },
                burn::tensor::ops::PadMode::Constant(0f32),
            );
        let unsqueeze206_out1 = [gather49_out1 as i64];
        let unsqueeze207_out1 = [gather51_out1 as i64];
        let constant886_out1: [i64; 1] = [10i64];
        let constant887_out1: [i64; 1] = [7i64];
        let constant888_out1: [i64; 1] = [10i64];
        let constant889_out1: [i64; 1] = [7i64];
        let concat108_out1: [i64; 6usize] = [
            &unsqueeze206_out1[..],
            &constant886_out1[..],
            &constant887_out1[..],
            &constant888_out1[..],
            &constant889_out1[..],
            &unsqueeze207_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape119_out1 = pad11_out1.reshape(concat108_out1);
        let transpose90_out1 = reshape119_out1.permute([0, 1, 3, 2, 4, 5]);
        let constant891_out1 = 10i64;
        let mul72_out1 = gather49_out1 * constant891_out1;
        let constant892_out1 = 10i64;
        let mul73_out1 = mul72_out1 * constant892_out1;
        let unsqueeze208_out1 = [mul73_out1 as i64];
        let unsqueeze209_out1 = [gather51_out1 as i64];
        let constant894_out1: [i64; 1] = [49i64];
        let concat109_out1: [i64; 3usize] = [
            &unsqueeze208_out1[..],
            &constant894_out1[..],
            &unsqueeze209_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape120_out1 = transpose90_out1.reshape(concat109_out1);
        let shape55_out1: [i64; 3] = {
            let axes = &reshape120_out1.clone().dims()[0..3];
            let mut output = [0i64; 3];
            for i in 0..3 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let gather52_out1 = shape55_out1[0] as i64;
        let gather53_out1 = shape55_out1[1] as i64;
        let layernormalization19_out1 = {
            let dtype = reshape120_out1.dtype();
            self.layernormalization19
                .forward(reshape120_out1.cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let linear37_out1 = self.linear37.forward(layernormalization19_out1);
        let unsqueeze210_out1 = [gather52_out1 as i64];
        let unsqueeze211_out1 = [gather53_out1 as i64];
        let constant901_out1: [i64; 1] = [-1i64];
        let constant900_out1: [i64; 1] = [10i64];
        let concat110_out1: [i64; 4usize] = [
            &unsqueeze210_out1[..],
            &unsqueeze211_out1[..],
            &constant900_out1[..],
            &constant901_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape121_out1 = linear37_out1.reshape(concat110_out1);
        let split_tensors = reshape121_out1.split_with_sizes([32, 32, 32].into(), 3);
        let [split10_out1, split10_out2, split10_out3] = split_tensors
            .try_into()
            .unwrap();
        let transpose91_out1 = split10_out1.permute([0, 2, 1, 3]);
        let transpose92_out1 = split10_out3.permute([0, 2, 1, 3]);
        let transpose93_out1 = split10_out2.permute([0, 2, 3, 1]);
        let matmul56_out1 = transpose91_out1.matmul(transpose93_out1);
        let constant902_out1 = self.constant902.val();
        let mul74_out1 = matmul56_out1
            .mul((constant902_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let constant903_out1 = self.constant903.val();
        let add52_out1 = mul74_out1.add((constant903_out1).unsqueeze_dims(&[0isize]));
        let softmax10_out1 = burn::tensor::activation::softmax(add52_out1, 3);
        let matmul57_out1 = softmax10_out1.matmul(transpose92_out1);
        let transpose94_out1 = matmul57_out1.permute([0, 2, 1, 3]);
        let unsqueeze212_out1 = [gather52_out1 as i64];
        let unsqueeze213_out1 = [gather53_out1 as i64];
        let constant906_out1: [i64; 1] = [320i64];
        let concat111_out1: [i64; 3usize] = [
            &unsqueeze212_out1[..],
            &unsqueeze213_out1[..],
            &constant906_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape122_out1 = transpose94_out1.reshape(concat111_out1);
        let linear38_out1 = self.linear38.forward(reshape122_out1);
        let unsqueeze214_out1 = [gather49_out1 as i64];
        let unsqueeze215_out1 = [gather51_out1 as i64];
        let constant908_out1: [i64; 1] = [10i64];
        let constant909_out1: [i64; 1] = [10i64];
        let constant910_out1: [i64; 1] = [7i64];
        let constant911_out1: [i64; 1] = [7i64];
        let concat112_out1: [i64; 6usize] = [
            &unsqueeze214_out1[..],
            &constant908_out1[..],
            &constant909_out1[..],
            &constant910_out1[..],
            &constant911_out1[..],
            &unsqueeze215_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape123_out1 = linear38_out1.reshape(concat112_out1);
        let transpose95_out1 = reshape123_out1.permute([0, 1, 3, 2, 4, 5]);
        let unsqueeze216_out1 = [gather49_out1 as i64];
        let unsqueeze217_out1 = [gather51_out1 as i64];
        let constant914_out1: [i64; 1] = [70i64];
        let constant915_out1: [i64; 1] = [70i64];
        let concat113_out1: [i64; 4usize] = [
            &unsqueeze216_out1[..],
            &constant914_out1[..],
            &constant915_out1[..],
            &unsqueeze217_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape124_out1 = transpose95_out1.reshape(concat113_out1);
        let slice33_out1 = reshape124_out1.slice(s![.., 0..64, .., ..]);
        let slice34_out1 = slice33_out1.slice(s![.., .., 0..64, ..]);
        let unsqueeze218_out1 = [gather49_out1 as i64];
        let unsqueeze219_out1 = [gather50_out1 as i64];
        let unsqueeze220_out1 = [gather51_out1 as i64];
        let concat114_out1: [i64; 3usize] = [
            &unsqueeze218_out1[..],
            &unsqueeze219_out1[..],
            &unsqueeze220_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape125_out1 = slice34_out1.reshape(concat114_out1);
        let add53_out1 = add51_out1.add(reshape125_out1);
        (add53_out1, gather49_out1, gather51_out1, gather50_out1)
    }
}
#[derive(Module, Debug)]
pub struct Submodule12 {
    conv2d27: Conv2d,
    layernormalization20: LayerNorm,
    linear39: Linear,
    constant935: burn::module::Param<Tensor<1>>,
    constant936: burn::module::Param<Tensor<1>>,
    constant937: burn::module::Param<Tensor<1>>,
    linear40: Linear,
    conv2d28: Conv2d,
    constant944: burn::module::Param<Tensor<1>>,
    constant945: burn::module::Param<Tensor<1>>,
    constant177: burn::module::Param<Tensor<3>>,
    constant178: burn::module::Param<Tensor<3>>,
    conv2d29: Conv2d,
    constant946: burn::module::Param<Tensor<1>>,
    constant947: burn::module::Param<Tensor<1>>,
    constant179: burn::module::Param<Tensor<3>>,
    constant180: burn::module::Param<Tensor<3>>,
    #[module(skip)]
    device: Device,
}
impl Submodule12 {
    #[allow(unused_variables)]
    pub fn new(device: &Device) -> Self {
        let conv2d27 = Conv2dConfig::new([320, 320], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(320)
            .with_bias(true)
            .init(device);
        let layernormalization20 = LayerNormConfig::new(320)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(true)
            .init(device);
        let linear39 = LinearConfig::new(320, 1280).with_bias(true).init(device);
        let constant935: burn::module::Param<Tensor<1>> = burn::module::Param::uninitialized(
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
        let constant936: burn::module::Param<Tensor<1>> = burn::module::Param::uninitialized(
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
        let constant937: burn::module::Param<Tensor<1>> = burn::module::Param::uninitialized(
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
        let linear40 = LinearConfig::new(1280, 320).with_bias(true).init(device);
        let conv2d28 = Conv2dConfig::new([320, 256], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let constant944: burn::module::Param<Tensor<1>> = burn::module::Param::uninitialized(
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
        let constant945: burn::module::Param<Tensor<1>> = burn::module::Param::uninitialized(
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
        let constant177: burn::module::Param<Tensor<3>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<
                3,
            >::zeros([256, 1, 1], (device, burn::tensor::DType::F32)),
            device.clone(),
            false,
            [256, 1, 1].into(),
        );
        let constant178: burn::module::Param<Tensor<3>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<
                3,
            >::zeros([256, 1, 1], (device, burn::tensor::DType::F32)),
            device.clone(),
            false,
            [256, 1, 1].into(),
        );
        let conv2d29 = Conv2dConfig::new([256, 256], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let constant946: burn::module::Param<Tensor<1>> = burn::module::Param::uninitialized(
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
        let constant947: burn::module::Param<Tensor<1>> = burn::module::Param::uninitialized(
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
        let constant179: burn::module::Param<Tensor<3>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<
                3,
            >::zeros([256, 1, 1], (device, burn::tensor::DType::F32)),
            device.clone(),
            false,
            [256, 1, 1].into(),
        );
        let constant180: burn::module::Param<Tensor<3>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<
                3,
            >::zeros([256, 1, 1], (device, burn::tensor::DType::F32)),
            device.clone(),
            false,
            [256, 1, 1].into(),
        );
        Self {
            conv2d27,
            layernormalization20,
            linear39,
            constant935,
            constant936,
            constant937,
            linear40,
            conv2d28,
            constant944,
            constant945,
            constant177,
            constant178,
            conv2d29,
            constant946,
            constant947,
            constant179,
            constant180,
            device: device.clone(),
        }
    }
    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(
        &self,
        add53_out1: Tensor<3>,
        gather49_out1: i64,
        gather51_out1: i64,
        gather50_out1: i64,
    ) -> Tensor<4> {
        let transpose96_out1 = add53_out1.permute([0, 2, 1]);
        let unsqueeze221_out1 = [gather49_out1 as i64];
        let unsqueeze222_out1 = [gather51_out1 as i64];
        let constant931_out1: [i64; 1] = [64i64];
        let constant930_out1: [i64; 1] = [64i64];
        let concat115_out1: [i64; 4usize] = [
            &unsqueeze221_out1[..],
            &unsqueeze222_out1[..],
            &constant930_out1[..],
            &constant931_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape126_out1 = transpose96_out1.reshape(concat115_out1);
        let conv2d27_out1 = self.conv2d27.forward(reshape126_out1);
        let unsqueeze223_out1 = [gather49_out1 as i64];
        let unsqueeze224_out1 = [gather51_out1 as i64];
        let unsqueeze225_out1 = [gather50_out1 as i64];
        let concat116_out1: [i64; 3usize] = [
            &unsqueeze223_out1[..],
            &unsqueeze224_out1[..],
            &unsqueeze225_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape127_out1 = conv2d27_out1.reshape(concat116_out1);
        let transpose97_out1 = reshape127_out1.permute([0, 2, 1]);
        let layernormalization20_out1 = {
            let dtype = transpose97_out1.clone().dtype();
            self.layernormalization20
                .forward(transpose97_out1.clone().cast(burn::tensor::DType::F32))
                .cast(dtype)
        };
        let linear39_out1 = self.linear39.forward(layernormalization20_out1);
        let constant935_out1 = self.constant935.val();
        let div24_out1 = linear39_out1
            .clone()
            .div((constant935_out1).unsqueeze_dims(&[0isize, 1isize]));
        let erf23_out1 = div24_out1.erf();
        let constant936_out1 = self.constant936.val();
        let add54_out1 = erf23_out1
            .add((constant936_out1).unsqueeze_dims(&[0isize, 1isize]));
        let mul75_out1 = linear39_out1.mul(add54_out1);
        let constant937_out1 = self.constant937.val();
        let mul76_out1 = mul75_out1
            .mul((constant937_out1).unsqueeze_dims(&[0isize, 1isize]));
        let linear40_out1 = self.linear40.forward(mul76_out1);
        let add55_out1 = transpose97_out1.add(linear40_out1);
        let shape57_out1: [i64; 3] = {
            let axes = &add55_out1.clone().dims()[0..3];
            let mut output = [0i64; 3];
            for i in 0..3 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let gather54_out1 = shape57_out1[0] as i64;
        let gather55_out1 = shape57_out1[2] as i64;
        let unsqueeze226_out1 = [gather54_out1 as i64];
        let unsqueeze227_out1 = [gather55_out1 as i64];
        let constant941_out1: [i64; 1] = [64i64];
        let constant942_out1: [i64; 1] = [64i64];
        let concat117_out1: [i64; 4usize] = [
            &unsqueeze226_out1[..],
            &constant941_out1[..],
            &constant942_out1[..],
            &unsqueeze227_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape128_out1 = add55_out1.reshape(concat117_out1);
        let transpose98_out1 = reshape128_out1.permute([0, 3, 1, 2]);
        let conv2d28_out1 = self.conv2d28.forward(transpose98_out1);
        let reducemean1_out1 = { conv2d28_out1.clone().mean_dim(1usize) };
        let sub5_out1 = conv2d28_out1.sub(reducemean1_out1);
        let constant944_out1 = self.constant944.val();
        let pow1_out1 = sub5_out1
            .clone()
            .powf((constant944_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let reducemean2_out1 = { pow1_out1.mean_dim(1usize) };
        let constant945_out1 = self.constant945.val();
        let add56_out1 = reducemean2_out1
            .add((constant945_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let sqrt1_out1 = add56_out1.sqrt();
        let div25_out1 = sub5_out1.div(sqrt1_out1);
        let constant177_out1 = self.constant177.val();
        let mul77_out1 = (constant177_out1).unsqueeze_dims(&[0isize]).mul(div25_out1);
        let constant178_out1 = self.constant178.val();
        let add57_out1 = mul77_out1.add((constant178_out1).unsqueeze_dims(&[0isize]));
        let conv2d29_out1 = self.conv2d29.forward(add57_out1);
        let reducemean3_out1 = { conv2d29_out1.clone().mean_dim(1usize) };
        let sub6_out1 = conv2d29_out1.sub(reducemean3_out1);
        let constant946_out1 = self.constant946.val();
        let pow2_out1 = sub6_out1
            .clone()
            .powf((constant946_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let reducemean4_out1 = { pow2_out1.mean_dim(1usize) };
        let constant947_out1 = self.constant947.val();
        let add58_out1 = reducemean4_out1
            .add((constant947_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let sqrt2_out1 = add58_out1.sqrt();
        let div26_out1 = sub6_out1.div(sqrt2_out1);
        let constant179_out1 = self.constant179.val();
        let mul78_out1 = (constant179_out1).unsqueeze_dims(&[0isize]).mul(div26_out1);
        let constant180_out1 = self.constant180.val();
        let add59_out1 = mul78_out1.add((constant180_out1).unsqueeze_dims(&[0isize]));
        add59_out1
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
    submodule7: Submodule7,
    submodule8: Submodule8,
    submodule9: Submodule9,
    submodule10: Submodule10,
    submodule11: Submodule11,
    submodule12: Submodule12,
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
        let submodule7 = Submodule7::new(device);
        let submodule8 = Submodule8::new(device);
        let submodule9 = Submodule9::new(device);
        let submodule10 = Submodule10::new(device);
        let submodule11 = Submodule11::new(device);
        let submodule12 = Submodule12::new(device);
        Self {
            submodule1,
            submodule2,
            submodule3,
            submodule4,
            submodule5,
            submodule6,
            submodule7,
            submodule8,
            submodule9,
            submodule10,
            submodule11,
            submodule12,
            device: device.clone(),
        }
    }

    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(&self, input_image: Tensor<3>) -> Tensor<4> {
        let add4_out1 = self.submodule1.forward(input_image);
        let reshape15_out1 = self.submodule2.forward(add4_out1);
        let reshape27_out1 = self.submodule3.forward(reshape15_out1);
        let reshape41_out1 = self.submodule4.forward(reshape27_out1);
        let reshape53_out1 = self.submodule5.forward(reshape41_out1);
        let reshape65_out1 = self.submodule6.forward(reshape53_out1);
        let reshape77_out1 = self.submodule7.forward(reshape65_out1);
        let reshape89_out1 = self.submodule8.forward(reshape77_out1);
        let reshape101_out1 = self.submodule9.forward(reshape89_out1);
        let reshape115_out1 = self.submodule10.forward(reshape101_out1);
        let (add53_out1, gather49_out1, gather51_out1, gather50_out1) = self
            .submodule11
            .forward(reshape115_out1);
        let add59_out1 = self
            .submodule12
            .forward(add53_out1, gather49_out1, gather51_out1, gather50_out1);
        add59_out1
    }
}
