// Generated from ONNX "/var/folders/vq/1wh9t5d5487bq7cf3y400n4r0000gn/T/tmp.F97HjEf2AT/palm_detection_mediapipe_2023feb.onnx" by burn-onnx
extern crate alloc;
use burn::prelude::*;
use burn::nn::PRelu;
use burn::nn::PReluConfig;
use burn::nn::PaddingConfig2d;
use burn::nn::conv::Conv2d;
use burn::nn::conv::Conv2dConfig;
use burn::nn::pool::MaxPool2d;
use burn::nn::pool::MaxPool2dConfig;


#[derive(Module, Debug)]
pub struct Model {
    conv2d1: Conv2d,
    prelu1: PRelu,
    conv2d2: Conv2d,
    conv2d3: Conv2d,
    prelu2: PRelu,
    conv2d4: Conv2d,
    conv2d5: Conv2d,
    prelu3: PRelu,
    conv2d6: Conv2d,
    conv2d7: Conv2d,
    prelu4: PRelu,
    maxpool2d1: MaxPool2d,
    conv2d8: Conv2d,
    conv2d9: Conv2d,
    prelu5: PRelu,
    conv2d10: Conv2d,
    conv2d11: Conv2d,
    prelu6: PRelu,
    conv2d12: Conv2d,
    conv2d13: Conv2d,
    prelu7: PRelu,
    conv2d14: Conv2d,
    conv2d15: Conv2d,
    prelu8: PRelu,
    maxpool2d2: MaxPool2d,
    conv2d16: Conv2d,
    conv2d17: Conv2d,
    prelu9: PRelu,
    conv2d18: Conv2d,
    conv2d19: Conv2d,
    prelu10: PRelu,
    conv2d20: Conv2d,
    conv2d21: Conv2d,
    prelu11: PRelu,
    conv2d22: Conv2d,
    conv2d23: Conv2d,
    prelu12: PRelu,
    maxpool2d3: MaxPool2d,
    conv2d24: Conv2d,
    conv2d25: Conv2d,
    prelu13: PRelu,
    conv2d26: Conv2d,
    conv2d27: Conv2d,
    prelu14: PRelu,
    conv2d28: Conv2d,
    conv2d29: Conv2d,
    prelu15: PRelu,
    conv2d30: Conv2d,
    conv2d31: Conv2d,
    prelu16: PRelu,
    maxpool2d4: MaxPool2d,
    conv2d32: Conv2d,
    conv2d33: Conv2d,
    prelu17: PRelu,
    conv2d34: Conv2d,
    conv2d35: Conv2d,
    prelu18: PRelu,
    conv2d36: Conv2d,
    conv2d37: Conv2d,
    prelu19: PRelu,
    conv2d38: Conv2d,
    conv2d39: Conv2d,
    prelu20: PRelu,
    resize1: burn::nn::interpolate::Interpolate2d,
    conv2d40: Conv2d,
    prelu21: PRelu,
    conv2d41: Conv2d,
    conv2d42: Conv2d,
    prelu22: PRelu,
    conv2d43: Conv2d,
    conv2d44: Conv2d,
    prelu23: PRelu,
    conv2d45: Conv2d,
    conv2d46: Conv2d,
    resize2: burn::nn::interpolate::Interpolate2d,
    conv2d47: Conv2d,
    prelu24: PRelu,
    conv2d48: Conv2d,
    conv2d49: Conv2d,
    prelu25: PRelu,
    conv2d50: Conv2d,
    conv2d51: Conv2d,
    prelu26: PRelu,
    conv2d52: Conv2d,
    conv2d53: Conv2d,
    #[module(skip)]
    device: Device,
}

impl Model {
    #[allow(unused_variables)]
    pub fn new(device: &Device) -> Self {
        let conv2d1 = Conv2dConfig::new([3, 32], [5, 5])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 2, 2))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let prelu1 = PReluConfig::new().with_num_parameters(32).init(device);
        let conv2d2 = Conv2dConfig::new([32, 32], [5, 5])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(2, 2, 2, 2))
            .with_dilation([1, 1])
            .with_groups(32)
            .with_bias(true)
            .init(device);
        let conv2d3 = Conv2dConfig::new([32, 32], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let prelu2 = PReluConfig::new().with_num_parameters(32).init(device);
        let conv2d4 = Conv2dConfig::new([32, 32], [5, 5])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(2, 2, 2, 2))
            .with_dilation([1, 1])
            .with_groups(32)
            .with_bias(true)
            .init(device);
        let conv2d5 = Conv2dConfig::new([32, 32], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let prelu3 = PReluConfig::new().with_num_parameters(32).init(device);
        let conv2d6 = Conv2dConfig::new([32, 32], [5, 5])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(2, 2, 2, 2))
            .with_dilation([1, 1])
            .with_groups(32)
            .with_bias(true)
            .init(device);
        let conv2d7 = Conv2dConfig::new([32, 32], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let prelu4 = PReluConfig::new().with_num_parameters(32).init(device);
        let maxpool2d1 = MaxPool2dConfig::new([2, 2])
            .with_strides([2, 2])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_ceil_mode(false)
            .init();
        let conv2d8 = Conv2dConfig::new([32, 32], [5, 5])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 2, 2))
            .with_dilation([1, 1])
            .with_groups(32)
            .with_bias(true)
            .init(device);
        let conv2d9 = Conv2dConfig::new([32, 64], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let prelu5 = PReluConfig::new().with_num_parameters(64).init(device);
        let conv2d10 = Conv2dConfig::new([64, 64], [5, 5])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(2, 2, 2, 2))
            .with_dilation([1, 1])
            .with_groups(64)
            .with_bias(true)
            .init(device);
        let conv2d11 = Conv2dConfig::new([64, 64], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let prelu6 = PReluConfig::new().with_num_parameters(64).init(device);
        let conv2d12 = Conv2dConfig::new([64, 64], [5, 5])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(2, 2, 2, 2))
            .with_dilation([1, 1])
            .with_groups(64)
            .with_bias(true)
            .init(device);
        let conv2d13 = Conv2dConfig::new([64, 64], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let prelu7 = PReluConfig::new().with_num_parameters(64).init(device);
        let conv2d14 = Conv2dConfig::new([64, 64], [5, 5])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(2, 2, 2, 2))
            .with_dilation([1, 1])
            .with_groups(64)
            .with_bias(true)
            .init(device);
        let conv2d15 = Conv2dConfig::new([64, 64], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let prelu8 = PReluConfig::new().with_num_parameters(64).init(device);
        let maxpool2d2 = MaxPool2dConfig::new([2, 2])
            .with_strides([2, 2])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_ceil_mode(false)
            .init();
        let conv2d16 = Conv2dConfig::new([64, 64], [5, 5])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 2, 2))
            .with_dilation([1, 1])
            .with_groups(64)
            .with_bias(true)
            .init(device);
        let conv2d17 = Conv2dConfig::new([64, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let prelu9 = PReluConfig::new().with_num_parameters(128).init(device);
        let conv2d18 = Conv2dConfig::new([128, 128], [5, 5])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(2, 2, 2, 2))
            .with_dilation([1, 1])
            .with_groups(128)
            .with_bias(true)
            .init(device);
        let conv2d19 = Conv2dConfig::new([128, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let prelu10 = PReluConfig::new().with_num_parameters(128).init(device);
        let conv2d20 = Conv2dConfig::new([128, 128], [5, 5])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(2, 2, 2, 2))
            .with_dilation([1, 1])
            .with_groups(128)
            .with_bias(true)
            .init(device);
        let conv2d21 = Conv2dConfig::new([128, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let prelu11 = PReluConfig::new().with_num_parameters(128).init(device);
        let conv2d22 = Conv2dConfig::new([128, 128], [5, 5])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(2, 2, 2, 2))
            .with_dilation([1, 1])
            .with_groups(128)
            .with_bias(true)
            .init(device);
        let conv2d23 = Conv2dConfig::new([128, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let prelu12 = PReluConfig::new().with_num_parameters(128).init(device);
        let maxpool2d3 = MaxPool2dConfig::new([2, 2])
            .with_strides([2, 2])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_ceil_mode(false)
            .init();
        let conv2d24 = Conv2dConfig::new([128, 128], [5, 5])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 2, 2))
            .with_dilation([1, 1])
            .with_groups(128)
            .with_bias(true)
            .init(device);
        let conv2d25 = Conv2dConfig::new([128, 256], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let prelu13 = PReluConfig::new().with_num_parameters(256).init(device);
        let conv2d26 = Conv2dConfig::new([256, 256], [5, 5])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(2, 2, 2, 2))
            .with_dilation([1, 1])
            .with_groups(256)
            .with_bias(true)
            .init(device);
        let conv2d27 = Conv2dConfig::new([256, 256], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let prelu14 = PReluConfig::new().with_num_parameters(256).init(device);
        let conv2d28 = Conv2dConfig::new([256, 256], [5, 5])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(2, 2, 2, 2))
            .with_dilation([1, 1])
            .with_groups(256)
            .with_bias(true)
            .init(device);
        let conv2d29 = Conv2dConfig::new([256, 256], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let prelu15 = PReluConfig::new().with_num_parameters(256).init(device);
        let conv2d30 = Conv2dConfig::new([256, 256], [5, 5])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(2, 2, 2, 2))
            .with_dilation([1, 1])
            .with_groups(256)
            .with_bias(true)
            .init(device);
        let conv2d31 = Conv2dConfig::new([256, 256], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let prelu16 = PReluConfig::new().with_num_parameters(256).init(device);
        let maxpool2d4 = MaxPool2dConfig::new([2, 2])
            .with_strides([2, 2])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_ceil_mode(false)
            .init();
        let conv2d32 = Conv2dConfig::new([256, 256], [5, 5])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 2, 2))
            .with_dilation([1, 1])
            .with_groups(256)
            .with_bias(true)
            .init(device);
        let conv2d33 = Conv2dConfig::new([256, 256], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let prelu17 = PReluConfig::new().with_num_parameters(256).init(device);
        let conv2d34 = Conv2dConfig::new([256, 256], [5, 5])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(2, 2, 2, 2))
            .with_dilation([1, 1])
            .with_groups(256)
            .with_bias(true)
            .init(device);
        let conv2d35 = Conv2dConfig::new([256, 256], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let prelu18 = PReluConfig::new().with_num_parameters(256).init(device);
        let conv2d36 = Conv2dConfig::new([256, 256], [5, 5])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(2, 2, 2, 2))
            .with_dilation([1, 1])
            .with_groups(256)
            .with_bias(true)
            .init(device);
        let conv2d37 = Conv2dConfig::new([256, 256], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let prelu19 = PReluConfig::new().with_num_parameters(256).init(device);
        let conv2d38 = Conv2dConfig::new([256, 256], [5, 5])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(2, 2, 2, 2))
            .with_dilation([1, 1])
            .with_groups(256)
            .with_bias(true)
            .init(device);
        let conv2d39 = Conv2dConfig::new([256, 256], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let prelu20 = PReluConfig::new().with_num_parameters(256).init(device);
        let resize1 = burn::nn::interpolate::Interpolate2dConfig::new()
            .with_output_size(Some([12, 12]))
            .with_scale_factor(None)
            .with_mode(burn::nn::interpolate::InterpolateMode::Linear)
            .with_align_corners(false)
            .init();
        let conv2d40 = Conv2dConfig::new([256, 256], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let prelu21 = PReluConfig::new().with_num_parameters(256).init(device);
        let conv2d41 = Conv2dConfig::new([256, 256], [5, 5])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(2, 2, 2, 2))
            .with_dilation([1, 1])
            .with_groups(256)
            .with_bias(true)
            .init(device);
        let conv2d42 = Conv2dConfig::new([256, 256], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let prelu22 = PReluConfig::new().with_num_parameters(256).init(device);
        let conv2d43 = Conv2dConfig::new([256, 256], [5, 5])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(2, 2, 2, 2))
            .with_dilation([1, 1])
            .with_groups(256)
            .with_bias(true)
            .init(device);
        let conv2d44 = Conv2dConfig::new([256, 256], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let prelu23 = PReluConfig::new().with_num_parameters(256).init(device);
        let conv2d45 = Conv2dConfig::new([256, 108], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d46 = Conv2dConfig::new([256, 6], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let resize2 = burn::nn::interpolate::Interpolate2dConfig::new()
            .with_output_size(Some([24, 24]))
            .with_scale_factor(None)
            .with_mode(burn::nn::interpolate::InterpolateMode::Linear)
            .with_align_corners(false)
            .init();
        let conv2d47 = Conv2dConfig::new([256, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let prelu24 = PReluConfig::new().with_num_parameters(128).init(device);
        let conv2d48 = Conv2dConfig::new([128, 128], [5, 5])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(2, 2, 2, 2))
            .with_dilation([1, 1])
            .with_groups(128)
            .with_bias(true)
            .init(device);
        let conv2d49 = Conv2dConfig::new([128, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let prelu25 = PReluConfig::new().with_num_parameters(128).init(device);
        let conv2d50 = Conv2dConfig::new([128, 128], [5, 5])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(2, 2, 2, 2))
            .with_dilation([1, 1])
            .with_groups(128)
            .with_bias(true)
            .init(device);
        let conv2d51 = Conv2dConfig::new([128, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let prelu26 = PReluConfig::new().with_num_parameters(128).init(device);
        let conv2d52 = Conv2dConfig::new([128, 36], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d53 = Conv2dConfig::new([128, 2], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        Self {
            conv2d1,
            prelu1,
            conv2d2,
            conv2d3,
            prelu2,
            conv2d4,
            conv2d5,
            prelu3,
            conv2d6,
            conv2d7,
            prelu4,
            maxpool2d1,
            conv2d8,
            conv2d9,
            prelu5,
            conv2d10,
            conv2d11,
            prelu6,
            conv2d12,
            conv2d13,
            prelu7,
            conv2d14,
            conv2d15,
            prelu8,
            maxpool2d2,
            conv2d16,
            conv2d17,
            prelu9,
            conv2d18,
            conv2d19,
            prelu10,
            conv2d20,
            conv2d21,
            prelu11,
            conv2d22,
            conv2d23,
            prelu12,
            maxpool2d3,
            conv2d24,
            conv2d25,
            prelu13,
            conv2d26,
            conv2d27,
            prelu14,
            conv2d28,
            conv2d29,
            prelu15,
            conv2d30,
            conv2d31,
            prelu16,
            maxpool2d4,
            conv2d32,
            conv2d33,
            prelu17,
            conv2d34,
            conv2d35,
            prelu18,
            conv2d36,
            conv2d37,
            prelu19,
            conv2d38,
            conv2d39,
            prelu20,
            resize1,
            conv2d40,
            prelu21,
            conv2d41,
            conv2d42,
            prelu22,
            conv2d43,
            conv2d44,
            prelu23,
            conv2d45,
            conv2d46,
            resize2,
            conv2d47,
            prelu24,
            conv2d48,
            conv2d49,
            prelu25,
            conv2d50,
            conv2d51,
            prelu26,
            conv2d52,
            conv2d53,
            device: device.clone(),
        }
    }

    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(&self, input_1: Tensor<4>) -> (Tensor<3>, Tensor<3>) {
        let transpose1_out1 = input_1.permute([0, 3, 1, 2]);
        let conv2d1_out1 = self.conv2d1.forward(transpose1_out1);
        let prelu1_out1 = self.prelu1.forward(conv2d1_out1);
        let conv2d2_out1 = self.conv2d2.forward(prelu1_out1.clone());
        let conv2d3_out1 = self.conv2d3.forward(conv2d2_out1);
        let add1_out1 = prelu1_out1.add(conv2d3_out1);
        let prelu2_out1 = self.prelu2.forward(add1_out1);
        let conv2d4_out1 = self.conv2d4.forward(prelu2_out1.clone());
        let conv2d5_out1 = self.conv2d5.forward(conv2d4_out1);
        let add2_out1 = prelu2_out1.add(conv2d5_out1);
        let prelu3_out1 = self.prelu3.forward(add2_out1);
        let conv2d6_out1 = self.conv2d6.forward(prelu3_out1.clone());
        let conv2d7_out1 = self.conv2d7.forward(conv2d6_out1);
        let add3_out1 = prelu3_out1.add(conv2d7_out1);
        let prelu4_out1 = self.prelu4.forward(add3_out1);
        let maxpool2d1_out1 = self.maxpool2d1.forward(prelu4_out1.clone());
        let pad1_out1 = maxpool2d1_out1
            .pad(
                [
                    (0usize, 0usize),
                    (0usize, 32usize),
                    (0usize, 0usize),
                    (0usize, 0usize),
                ],
                burn::tensor::ops::PadMode::Constant(0f32),
            );
        let conv2d8_out1 = self.conv2d8.forward(prelu4_out1);
        let conv2d9_out1 = self.conv2d9.forward(conv2d8_out1);
        let add4_out1 = pad1_out1.add(conv2d9_out1);
        let prelu5_out1 = self.prelu5.forward(add4_out1);
        let conv2d10_out1 = self.conv2d10.forward(prelu5_out1.clone());
        let conv2d11_out1 = self.conv2d11.forward(conv2d10_out1);
        let add5_out1 = prelu5_out1.add(conv2d11_out1);
        let prelu6_out1 = self.prelu6.forward(add5_out1);
        let conv2d12_out1 = self.conv2d12.forward(prelu6_out1.clone());
        let conv2d13_out1 = self.conv2d13.forward(conv2d12_out1);
        let add6_out1 = prelu6_out1.add(conv2d13_out1);
        let prelu7_out1 = self.prelu7.forward(add6_out1);
        let conv2d14_out1 = self.conv2d14.forward(prelu7_out1.clone());
        let conv2d15_out1 = self.conv2d15.forward(conv2d14_out1);
        let add7_out1 = prelu7_out1.add(conv2d15_out1);
        let prelu8_out1 = self.prelu8.forward(add7_out1);
        let maxpool2d2_out1 = self.maxpool2d2.forward(prelu8_out1.clone());
        let pad2_out1 = maxpool2d2_out1
            .pad(
                [
                    (0usize, 0usize),
                    (0usize, 64usize),
                    (0usize, 0usize),
                    (0usize, 0usize),
                ],
                burn::tensor::ops::PadMode::Constant(0f32),
            );
        let conv2d16_out1 = self.conv2d16.forward(prelu8_out1);
        let conv2d17_out1 = self.conv2d17.forward(conv2d16_out1);
        let add8_out1 = pad2_out1.add(conv2d17_out1);
        let prelu9_out1 = self.prelu9.forward(add8_out1);
        let conv2d18_out1 = self.conv2d18.forward(prelu9_out1.clone());
        let conv2d19_out1 = self.conv2d19.forward(conv2d18_out1);
        let add9_out1 = prelu9_out1.add(conv2d19_out1);
        let prelu10_out1 = self.prelu10.forward(add9_out1);
        let conv2d20_out1 = self.conv2d20.forward(prelu10_out1.clone());
        let conv2d21_out1 = self.conv2d21.forward(conv2d20_out1);
        let add10_out1 = prelu10_out1.add(conv2d21_out1);
        let prelu11_out1 = self.prelu11.forward(add10_out1);
        let conv2d22_out1 = self.conv2d22.forward(prelu11_out1.clone());
        let conv2d23_out1 = self.conv2d23.forward(conv2d22_out1);
        let add11_out1 = prelu11_out1.add(conv2d23_out1);
        let prelu12_out1 = self.prelu12.forward(add11_out1);
        let maxpool2d3_out1 = self.maxpool2d3.forward(prelu12_out1.clone());
        let pad3_out1 = maxpool2d3_out1
            .pad(
                [
                    (0usize, 0usize),
                    (0usize, 128usize),
                    (0usize, 0usize),
                    (0usize, 0usize),
                ],
                burn::tensor::ops::PadMode::Constant(0f32),
            );
        let conv2d24_out1 = self.conv2d24.forward(prelu12_out1.clone());
        let conv2d25_out1 = self.conv2d25.forward(conv2d24_out1);
        let add12_out1 = pad3_out1.add(conv2d25_out1);
        let prelu13_out1 = self.prelu13.forward(add12_out1);
        let conv2d26_out1 = self.conv2d26.forward(prelu13_out1.clone());
        let conv2d27_out1 = self.conv2d27.forward(conv2d26_out1);
        let add13_out1 = prelu13_out1.add(conv2d27_out1);
        let prelu14_out1 = self.prelu14.forward(add13_out1);
        let conv2d28_out1 = self.conv2d28.forward(prelu14_out1.clone());
        let conv2d29_out1 = self.conv2d29.forward(conv2d28_out1);
        let add14_out1 = prelu14_out1.add(conv2d29_out1);
        let prelu15_out1 = self.prelu15.forward(add14_out1);
        let conv2d30_out1 = self.conv2d30.forward(prelu15_out1.clone());
        let conv2d31_out1 = self.conv2d31.forward(conv2d30_out1);
        let add15_out1 = prelu15_out1.add(conv2d31_out1);
        let prelu16_out1 = self.prelu16.forward(add15_out1);
        let maxpool2d4_out1 = self.maxpool2d4.forward(prelu16_out1.clone());
        let conv2d32_out1 = self.conv2d32.forward(prelu16_out1.clone());
        let conv2d33_out1 = self.conv2d33.forward(conv2d32_out1);
        let add16_out1 = maxpool2d4_out1.add(conv2d33_out1);
        let prelu17_out1 = self.prelu17.forward(add16_out1);
        let conv2d34_out1 = self.conv2d34.forward(prelu17_out1.clone());
        let conv2d35_out1 = self.conv2d35.forward(conv2d34_out1);
        let add17_out1 = prelu17_out1.add(conv2d35_out1);
        let prelu18_out1 = self.prelu18.forward(add17_out1);
        let conv2d36_out1 = self.conv2d36.forward(prelu18_out1.clone());
        let conv2d37_out1 = self.conv2d37.forward(conv2d36_out1);
        let add18_out1 = prelu18_out1.add(conv2d37_out1);
        let prelu19_out1 = self.prelu19.forward(add18_out1);
        let conv2d38_out1 = self.conv2d38.forward(prelu19_out1.clone());
        let conv2d39_out1 = self.conv2d39.forward(conv2d38_out1);
        let add19_out1 = prelu19_out1.add(conv2d39_out1);
        let prelu20_out1 = self.prelu20.forward(add19_out1);
        let resize1_out1 = self.resize1.forward(prelu20_out1);
        let conv2d40_out1 = self.conv2d40.forward(resize1_out1);
        let prelu21_out1 = self.prelu21.forward(conv2d40_out1);
        let add20_out1 = prelu16_out1.add(prelu21_out1);
        let conv2d41_out1 = self.conv2d41.forward(add20_out1.clone());
        let conv2d42_out1 = self.conv2d42.forward(conv2d41_out1);
        let add21_out1 = add20_out1.add(conv2d42_out1);
        let prelu22_out1 = self.prelu22.forward(add21_out1);
        let conv2d43_out1 = self.conv2d43.forward(prelu22_out1.clone());
        let conv2d44_out1 = self.conv2d44.forward(conv2d43_out1);
        let add22_out1 = prelu22_out1.add(conv2d44_out1);
        let prelu23_out1 = self.prelu23.forward(add22_out1);
        let conv2d45_out1 = self.conv2d45.forward(prelu23_out1.clone());
        let transpose2_out1 = conv2d45_out1.permute([0, 2, 3, 1]);
        let reshape1_out1 = transpose2_out1.reshape([1, -1, 18]);
        let conv2d46_out1 = self.conv2d46.forward(prelu23_out1.clone());
        let transpose3_out1 = conv2d46_out1.permute([0, 2, 3, 1]);
        let reshape2_out1 = transpose3_out1.reshape([1, -1, 1]);
        let resize2_out1 = self.resize2.forward(prelu23_out1);
        let conv2d47_out1 = self.conv2d47.forward(resize2_out1);
        let prelu24_out1 = self.prelu24.forward(conv2d47_out1);
        let add23_out1 = prelu12_out1.add(prelu24_out1);
        let conv2d48_out1 = self.conv2d48.forward(add23_out1.clone());
        let conv2d49_out1 = self.conv2d49.forward(conv2d48_out1);
        let add24_out1 = add23_out1.add(conv2d49_out1);
        let prelu25_out1 = self.prelu25.forward(add24_out1);
        let conv2d50_out1 = self.conv2d50.forward(prelu25_out1.clone());
        let conv2d51_out1 = self.conv2d51.forward(conv2d50_out1);
        let add25_out1 = prelu25_out1.add(conv2d51_out1);
        let prelu26_out1 = self.prelu26.forward(add25_out1);
        let conv2d52_out1 = self.conv2d52.forward(prelu26_out1.clone());
        let transpose4_out1 = conv2d52_out1.permute([0, 2, 3, 1]);
        let reshape3_out1 = transpose4_out1.reshape([1, -1, 18]);
        let concat1_out1 = burn::tensor::Tensor::cat(
            [reshape3_out1, reshape1_out1].into(),
            1,
        );
        let conv2d53_out1 = self.conv2d53.forward(prelu26_out1);
        let transpose5_out1 = conv2d53_out1.permute([0, 2, 3, 1]);
        let reshape4_out1 = transpose5_out1.reshape([1, -1, 1]);
        let concat2_out1 = burn::tensor::Tensor::cat(
            [reshape4_out1, reshape2_out1].into(),
            1,
        );
        (concat1_out1, concat2_out1)
    }
}
