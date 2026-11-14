// Generated from ONNX "/var/folders/vq/1wh9t5d5487bq7cf3y400n4r0000gn/T/tmp.F97HjEf2AT/person_detection_mediapipe_2023mar.onnx" by burn-onnx
extern crate alloc;
use burn::prelude::*;
use burn::nn::PaddingConfig2d;
use burn::nn::conv::Conv2d;
use burn::nn::conv::Conv2dConfig;


#[derive(Module, Debug)]
pub struct Model {
    conv2d1: Conv2d,
    conv2d2: Conv2d,
    conv2d3: Conv2d,
    conv2d4: Conv2d,
    conv2d5: Conv2d,
    conv2d6: Conv2d,
    conv2d7: Conv2d,
    conv2d8: Conv2d,
    conv2d9: Conv2d,
    conv2d10: Conv2d,
    conv2d11: Conv2d,
    conv2d12: Conv2d,
    conv2d13: Conv2d,
    conv2d14: Conv2d,
    conv2d15: Conv2d,
    conv2d16: Conv2d,
    conv2d17: Conv2d,
    conv2d18: Conv2d,
    conv2d19: Conv2d,
    conv2d20: Conv2d,
    conv2d21: Conv2d,
    conv2d22: Conv2d,
    conv2d23: Conv2d,
    conv2d24: Conv2d,
    conv2d25: Conv2d,
    conv2d26: Conv2d,
    conv2d27: Conv2d,
    conv2d28: Conv2d,
    conv2d29: Conv2d,
    conv2d30: Conv2d,
    conv2d31: Conv2d,
    conv2d32: Conv2d,
    conv2d33: Conv2d,
    conv2d34: Conv2d,
    conv2d35: Conv2d,
    conv2d36: Conv2d,
    conv2d37: Conv2d,
    conv2d38: Conv2d,
    conv2d39: Conv2d,
    conv2d40: Conv2d,
    conv2d41: Conv2d,
    conv2d42: Conv2d,
    conv2d43: Conv2d,
    conv2d44: Conv2d,
    conv2d45: Conv2d,
    conv2d46: Conv2d,
    conv2d47: Conv2d,
    conv2d48: Conv2d,
    conv2d49: Conv2d,
    conv2d50: Conv2d,
    conv2d51: Conv2d,
    resize1: burn::nn::interpolate::Interpolate2d,
    conv2d52: Conv2d,
    conv2d53: Conv2d,
    conv2d54: Conv2d,
    conv2d55: Conv2d,
    conv2d56: Conv2d,
    conv2d57: Conv2d,
    conv2d58: Conv2d,
    conv2d59: Conv2d,
    conv2d60: Conv2d,
    conv2d61: Conv2d,
    conv2d62: Conv2d,
    resize2: burn::nn::interpolate::Interpolate2d,
    conv2d63: Conv2d,
    conv2d64: Conv2d,
    conv2d65: Conv2d,
    conv2d66: Conv2d,
    conv2d67: Conv2d,
    conv2d68: Conv2d,
    conv2d69: Conv2d,
    conv2d70: Conv2d,
    conv2d71: Conv2d,
    conv2d72: Conv2d,
    conv2d73: Conv2d,
    #[module(skip)]
    device: Device,
}

impl Model {
    #[allow(unused_variables)]
    pub fn new(device: &Device) -> Self {
        let conv2d1 = Conv2dConfig::new([3, 24], [3, 3])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d2 = Conv2dConfig::new([24, 24], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(24)
            .with_bias(true)
            .init(device);
        let conv2d3 = Conv2dConfig::new([24, 16], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d4 = Conv2dConfig::new([16, 64], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d5 = Conv2dConfig::new([64, 64], [3, 3])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(64)
            .with_bias(true)
            .init(device);
        let conv2d6 = Conv2dConfig::new([64, 24], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d7 = Conv2dConfig::new([24, 144], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d8 = Conv2dConfig::new([144, 144], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(144)
            .with_bias(true)
            .init(device);
        let conv2d9 = Conv2dConfig::new([144, 24], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d10 = Conv2dConfig::new([24, 144], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d11 = Conv2dConfig::new([144, 144], [5, 5])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Explicit(2, 2, 2, 2))
            .with_dilation([1, 1])
            .with_groups(144)
            .with_bias(true)
            .init(device);
        let conv2d12 = Conv2dConfig::new([144, 40], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d13 = Conv2dConfig::new([40, 240], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d14 = Conv2dConfig::new([240, 240], [5, 5])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(2, 2, 2, 2))
            .with_dilation([1, 1])
            .with_groups(240)
            .with_bias(true)
            .init(device);
        let conv2d15 = Conv2dConfig::new([240, 40], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d16 = Conv2dConfig::new([40, 240], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d17 = Conv2dConfig::new([240, 240], [3, 3])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(240)
            .with_bias(true)
            .init(device);
        let conv2d18 = Conv2dConfig::new([240, 80], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d19 = Conv2dConfig::new([80, 480], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d20 = Conv2dConfig::new([480, 480], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(480)
            .with_bias(true)
            .init(device);
        let conv2d21 = Conv2dConfig::new([480, 80], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d22 = Conv2dConfig::new([80, 480], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d23 = Conv2dConfig::new([480, 480], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(480)
            .with_bias(true)
            .init(device);
        let conv2d24 = Conv2dConfig::new([480, 80], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d25 = Conv2dConfig::new([80, 480], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d26 = Conv2dConfig::new([480, 480], [5, 5])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(2, 2, 2, 2))
            .with_dilation([1, 1])
            .with_groups(480)
            .with_bias(true)
            .init(device);
        let conv2d27 = Conv2dConfig::new([480, 112], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d28 = Conv2dConfig::new([112, 672], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d29 = Conv2dConfig::new([672, 672], [5, 5])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(2, 2, 2, 2))
            .with_dilation([1, 1])
            .with_groups(672)
            .with_bias(true)
            .init(device);
        let conv2d30 = Conv2dConfig::new([672, 112], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d31 = Conv2dConfig::new([112, 672], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d32 = Conv2dConfig::new([672, 672], [5, 5])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(2, 2, 2, 2))
            .with_dilation([1, 1])
            .with_groups(672)
            .with_bias(true)
            .init(device);
        let conv2d33 = Conv2dConfig::new([672, 112], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d34 = Conv2dConfig::new([112, 672], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d35 = Conv2dConfig::new([672, 672], [5, 5])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Explicit(2, 2, 2, 2))
            .with_dilation([1, 1])
            .with_groups(672)
            .with_bias(true)
            .init(device);
        let conv2d36 = Conv2dConfig::new([672, 192], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d37 = Conv2dConfig::new([192, 1152], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d38 = Conv2dConfig::new([1152, 1152], [5, 5])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(2, 2, 2, 2))
            .with_dilation([1, 1])
            .with_groups(1152)
            .with_bias(true)
            .init(device);
        let conv2d39 = Conv2dConfig::new([1152, 192], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d40 = Conv2dConfig::new([192, 1152], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d41 = Conv2dConfig::new([1152, 1152], [5, 5])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(2, 2, 2, 2))
            .with_dilation([1, 1])
            .with_groups(1152)
            .with_bias(true)
            .init(device);
        let conv2d42 = Conv2dConfig::new([1152, 192], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d43 = Conv2dConfig::new([192, 1152], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d44 = Conv2dConfig::new([1152, 1152], [5, 5])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(2, 2, 2, 2))
            .with_dilation([1, 1])
            .with_groups(1152)
            .with_bias(true)
            .init(device);
        let conv2d45 = Conv2dConfig::new([1152, 192], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d46 = Conv2dConfig::new([192, 1152], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d47 = Conv2dConfig::new([1152, 1152], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1152)
            .with_bias(true)
            .init(device);
        let conv2d48 = Conv2dConfig::new([288, 288], [3, 3])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(288)
            .with_bias(true)
            .init(device);
        let conv2d49 = Conv2dConfig::new([288, 72], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d50 = Conv2dConfig::new([288, 288], [3, 3])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(288)
            .with_bias(true)
            .init(device);
        let conv2d51 = Conv2dConfig::new([288, 6], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let resize1 = burn::nn::interpolate::Interpolate2dConfig::new()
            .with_output_size(Some([14, 14]))
            .with_scale_factor(None)
            .with_mode(burn::nn::interpolate::InterpolateMode::Linear)
            .with_align_corners(false)
            .init();
        let conv2d52 = Conv2dConfig::new([1152, 192], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d53 = Conv2dConfig::new([112, 112], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(112)
            .with_bias(true)
            .init(device);
        let conv2d54 = Conv2dConfig::new([112, 192], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d55 = Conv2dConfig::new([192, 192], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(192)
            .with_bias(true)
            .init(device);
        let conv2d56 = Conv2dConfig::new([192, 192], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d57 = Conv2dConfig::new([192, 192], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(192)
            .with_bias(true)
            .init(device);
        let conv2d58 = Conv2dConfig::new([192, 192], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d59 = Conv2dConfig::new([48, 48], [3, 3])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(48)
            .with_bias(true)
            .init(device);
        let conv2d60 = Conv2dConfig::new([48, 24], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d61 = Conv2dConfig::new([48, 48], [3, 3])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(48)
            .with_bias(true)
            .init(device);
        let conv2d62 = Conv2dConfig::new([48, 2], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let resize2 = burn::nn::interpolate::Interpolate2dConfig::new()
            .with_output_size(Some([28, 28]))
            .with_scale_factor(None)
            .with_mode(burn::nn::interpolate::InterpolateMode::Linear)
            .with_align_corners(false)
            .init();
        let conv2d63 = Conv2dConfig::new([192, 96], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d64 = Conv2dConfig::new([40, 40], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(40)
            .with_bias(true)
            .init(device);
        let conv2d65 = Conv2dConfig::new([40, 96], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d66 = Conv2dConfig::new([96, 96], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(96)
            .with_bias(true)
            .init(device);
        let conv2d67 = Conv2dConfig::new([96, 96], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d68 = Conv2dConfig::new([96, 96], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(96)
            .with_bias(true)
            .init(device);
        let conv2d69 = Conv2dConfig::new([96, 96], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d70 = Conv2dConfig::new([24, 24], [3, 3])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(24)
            .with_bias(true)
            .init(device);
        let conv2d71 = Conv2dConfig::new([24, 24], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d72 = Conv2dConfig::new([24, 24], [3, 3])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(24)
            .with_bias(true)
            .init(device);
        let conv2d73 = Conv2dConfig::new([24, 2], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        Self {
            conv2d1,
            conv2d2,
            conv2d3,
            conv2d4,
            conv2d5,
            conv2d6,
            conv2d7,
            conv2d8,
            conv2d9,
            conv2d10,
            conv2d11,
            conv2d12,
            conv2d13,
            conv2d14,
            conv2d15,
            conv2d16,
            conv2d17,
            conv2d18,
            conv2d19,
            conv2d20,
            conv2d21,
            conv2d22,
            conv2d23,
            conv2d24,
            conv2d25,
            conv2d26,
            conv2d27,
            conv2d28,
            conv2d29,
            conv2d30,
            conv2d31,
            conv2d32,
            conv2d33,
            conv2d34,
            conv2d35,
            conv2d36,
            conv2d37,
            conv2d38,
            conv2d39,
            conv2d40,
            conv2d41,
            conv2d42,
            conv2d43,
            conv2d44,
            conv2d45,
            conv2d46,
            conv2d47,
            conv2d48,
            conv2d49,
            conv2d50,
            conv2d51,
            resize1,
            conv2d52,
            conv2d53,
            conv2d54,
            conv2d55,
            conv2d56,
            conv2d57,
            conv2d58,
            conv2d59,
            conv2d60,
            conv2d61,
            conv2d62,
            resize2,
            conv2d63,
            conv2d64,
            conv2d65,
            conv2d66,
            conv2d67,
            conv2d68,
            conv2d69,
            conv2d70,
            conv2d71,
            conv2d72,
            conv2d73,
            device: device.clone(),
        }
    }

    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(&self, input_1_0: Tensor<4>) -> (Tensor<3>, Tensor<3>) {
        let conv2d1_out1 = self.conv2d1.forward(input_1_0);
        let clip1_out1 = conv2d1_out1.clamp(0f64, 6f64);
        let conv2d2_out1 = self.conv2d2.forward(clip1_out1);
        let clip2_out1 = conv2d2_out1.clamp(0f64, 6f64);
        let conv2d3_out1 = self.conv2d3.forward(clip2_out1);
        let conv2d4_out1 = self.conv2d4.forward(conv2d3_out1);
        let clip3_out1 = conv2d4_out1.clamp(0f64, 6f64);
        let conv2d5_out1 = self.conv2d5.forward(clip3_out1);
        let clip4_out1 = conv2d5_out1.clamp(0f64, 6f64);
        let conv2d6_out1 = self.conv2d6.forward(clip4_out1);
        let conv2d7_out1 = self.conv2d7.forward(conv2d6_out1.clone());
        let clip5_out1 = conv2d7_out1.clamp(0f64, 6f64);
        let conv2d8_out1 = self.conv2d8.forward(clip5_out1);
        let clip6_out1 = conv2d8_out1.clamp(0f64, 6f64);
        let conv2d9_out1 = self.conv2d9.forward(clip6_out1);
        let add1_out1 = conv2d9_out1.add(conv2d6_out1);
        let conv2d10_out1 = self.conv2d10.forward(add1_out1);
        let clip7_out1 = conv2d10_out1.clamp(0f64, 6f64);
        let conv2d11_out1 = self.conv2d11.forward(clip7_out1);
        let clip8_out1 = conv2d11_out1.clamp(0f64, 6f64);
        let conv2d12_out1 = self.conv2d12.forward(clip8_out1);
        let conv2d13_out1 = self.conv2d13.forward(conv2d12_out1.clone());
        let clip9_out1 = conv2d13_out1.clamp(0f64, 6f64);
        let conv2d14_out1 = self.conv2d14.forward(clip9_out1);
        let clip10_out1 = conv2d14_out1.clamp(0f64, 6f64);
        let conv2d15_out1 = self.conv2d15.forward(clip10_out1);
        let add2_out1 = conv2d15_out1.add(conv2d12_out1);
        let conv2d16_out1 = self.conv2d16.forward(add2_out1.clone());
        let clip11_out1 = conv2d16_out1.clamp(0f64, 6f64);
        let conv2d17_out1 = self.conv2d17.forward(clip11_out1);
        let clip12_out1 = conv2d17_out1.clamp(0f64, 6f64);
        let conv2d18_out1 = self.conv2d18.forward(clip12_out1);
        let conv2d19_out1 = self.conv2d19.forward(conv2d18_out1.clone());
        let clip13_out1 = conv2d19_out1.clamp(0f64, 6f64);
        let conv2d20_out1 = self.conv2d20.forward(clip13_out1);
        let clip14_out1 = conv2d20_out1.clamp(0f64, 6f64);
        let conv2d21_out1 = self.conv2d21.forward(clip14_out1);
        let add3_out1 = conv2d21_out1.add(conv2d18_out1);
        let conv2d22_out1 = self.conv2d22.forward(add3_out1.clone());
        let clip15_out1 = conv2d22_out1.clamp(0f64, 6f64);
        let conv2d23_out1 = self.conv2d23.forward(clip15_out1);
        let clip16_out1 = conv2d23_out1.clamp(0f64, 6f64);
        let conv2d24_out1 = self.conv2d24.forward(clip16_out1);
        let add4_out1 = conv2d24_out1.add(add3_out1);
        let conv2d25_out1 = self.conv2d25.forward(add4_out1);
        let clip17_out1 = conv2d25_out1.clamp(0f64, 6f64);
        let conv2d26_out1 = self.conv2d26.forward(clip17_out1);
        let clip18_out1 = conv2d26_out1.clamp(0f64, 6f64);
        let conv2d27_out1 = self.conv2d27.forward(clip18_out1);
        let conv2d28_out1 = self.conv2d28.forward(conv2d27_out1.clone());
        let clip19_out1 = conv2d28_out1.clamp(0f64, 6f64);
        let conv2d29_out1 = self.conv2d29.forward(clip19_out1);
        let clip20_out1 = conv2d29_out1.clamp(0f64, 6f64);
        let conv2d30_out1 = self.conv2d30.forward(clip20_out1);
        let add5_out1 = conv2d30_out1.add(conv2d27_out1);
        let conv2d31_out1 = self.conv2d31.forward(add5_out1.clone());
        let clip21_out1 = conv2d31_out1.clamp(0f64, 6f64);
        let conv2d32_out1 = self.conv2d32.forward(clip21_out1);
        let clip22_out1 = conv2d32_out1.clamp(0f64, 6f64);
        let conv2d33_out1 = self.conv2d33.forward(clip22_out1);
        let add6_out1 = conv2d33_out1.add(add5_out1);
        let conv2d34_out1 = self.conv2d34.forward(add6_out1.clone());
        let clip23_out1 = conv2d34_out1.clamp(0f64, 6f64);
        let conv2d35_out1 = self.conv2d35.forward(clip23_out1);
        let clip24_out1 = conv2d35_out1.clamp(0f64, 6f64);
        let conv2d36_out1 = self.conv2d36.forward(clip24_out1);
        let conv2d37_out1 = self.conv2d37.forward(conv2d36_out1.clone());
        let clip25_out1 = conv2d37_out1.clamp(0f64, 6f64);
        let conv2d38_out1 = self.conv2d38.forward(clip25_out1);
        let clip26_out1 = conv2d38_out1.clamp(0f64, 6f64);
        let conv2d39_out1 = self.conv2d39.forward(clip26_out1);
        let add7_out1 = conv2d39_out1.add(conv2d36_out1);
        let conv2d40_out1 = self.conv2d40.forward(add7_out1.clone());
        let clip27_out1 = conv2d40_out1.clamp(0f64, 6f64);
        let conv2d41_out1 = self.conv2d41.forward(clip27_out1);
        let clip28_out1 = conv2d41_out1.clamp(0f64, 6f64);
        let conv2d42_out1 = self.conv2d42.forward(clip28_out1);
        let add8_out1 = conv2d42_out1.add(add7_out1);
        let conv2d43_out1 = self.conv2d43.forward(add8_out1.clone());
        let clip29_out1 = conv2d43_out1.clamp(0f64, 6f64);
        let conv2d44_out1 = self.conv2d44.forward(clip29_out1);
        let clip30_out1 = conv2d44_out1.clamp(0f64, 6f64);
        let conv2d45_out1 = self.conv2d45.forward(clip30_out1);
        let add9_out1 = conv2d45_out1.add(add8_out1);
        let conv2d46_out1 = self.conv2d46.forward(add9_out1);
        let clip31_out1 = conv2d46_out1.clamp(0f64, 6f64);
        let conv2d47_out1 = self.conv2d47.forward(clip31_out1);
        let clip32_out1 = conv2d47_out1.clamp(0f64, 6f64);
        let depthtospace1_out1 = {
            let [b, c, h, w] = clip32_out1.clone().dims();
            let crd = clip32_out1
                .clone()
                .reshape([b, 2usize * 2usize, c / (2usize * 2usize), h, w])
                .swap_dims(1, 2)
                .reshape([b, c, h, w]);
            burn::nn::PixelShuffleConfig::new(2usize).init().forward(crd)
        };
        let pad1_out1 = depthtospace1_out1
            .pad(
                [(0usize, 0usize), (0usize, 0usize), (1usize, 1usize), (1usize, 1usize)],
                burn::tensor::ops::PadMode::Constant(0f32),
            );
        let conv2d48_out1 = self.conv2d48.forward(pad1_out1.clone());
        let conv2d49_out1 = self.conv2d49.forward(conv2d48_out1);
        let transpose1_out1 = conv2d49_out1.permute([0, 2, 3, 1]);
        let reshape1_out1 = transpose1_out1.reshape([1, -1, 12]);
        let conv2d50_out1 = self.conv2d50.forward(pad1_out1);
        let conv2d51_out1 = self.conv2d51.forward(conv2d50_out1);
        let transpose2_out1 = conv2d51_out1.permute([0, 2, 3, 1]);
        let reshape2_out1 = transpose2_out1.reshape([1, -1, 1]);
        let resize1_out1 = self.resize1.forward(clip32_out1);
        let conv2d52_out1 = self.conv2d52.forward(resize1_out1);
        let relu1_out1 = burn::tensor::activation::relu(conv2d52_out1);
        let conv2d53_out1 = self.conv2d53.forward(add6_out1);
        let conv2d54_out1 = self.conv2d54.forward(conv2d53_out1);
        let relu2_out1 = burn::tensor::activation::relu(conv2d54_out1);
        let add10_out1 = relu2_out1.add(relu1_out1);
        let conv2d55_out1 = self.conv2d55.forward(add10_out1.clone());
        let conv2d56_out1 = self.conv2d56.forward(conv2d55_out1);
        let add11_out1 = add10_out1.add(conv2d56_out1);
        let relu3_out1 = burn::tensor::activation::relu(add11_out1);
        let conv2d57_out1 = self.conv2d57.forward(relu3_out1.clone());
        let conv2d58_out1 = self.conv2d58.forward(conv2d57_out1);
        let add12_out1 = relu3_out1.add(conv2d58_out1);
        let relu4_out1 = burn::tensor::activation::relu(add12_out1);
        let depthtospace2_out1 = {
            let [b, c, h, w] = relu4_out1.clone().dims();
            let crd = relu4_out1
                .clone()
                .reshape([b, 2usize * 2usize, c / (2usize * 2usize), h, w])
                .swap_dims(1, 2)
                .reshape([b, c, h, w]);
            burn::nn::PixelShuffleConfig::new(2usize).init().forward(crd)
        };
        let pad2_out1 = depthtospace2_out1
            .pad(
                [(0usize, 0usize), (0usize, 0usize), (1usize, 1usize), (1usize, 1usize)],
                burn::tensor::ops::PadMode::Constant(0f32),
            );
        let conv2d59_out1 = self.conv2d59.forward(pad2_out1.clone());
        let conv2d60_out1 = self.conv2d60.forward(conv2d59_out1);
        let transpose3_out1 = conv2d60_out1.permute([0, 2, 3, 1]);
        let reshape3_out1 = transpose3_out1.reshape([1, -1, 12]);
        let conv2d61_out1 = self.conv2d61.forward(pad2_out1);
        let conv2d62_out1 = self.conv2d62.forward(conv2d61_out1);
        let transpose4_out1 = conv2d62_out1.permute([0, 2, 3, 1]);
        let reshape4_out1 = transpose4_out1.reshape([1, -1, 1]);
        let resize2_out1 = self.resize2.forward(relu4_out1);
        let conv2d63_out1 = self.conv2d63.forward(resize2_out1);
        let relu5_out1 = burn::tensor::activation::relu(conv2d63_out1);
        let conv2d64_out1 = self.conv2d64.forward(add2_out1);
        let conv2d65_out1 = self.conv2d65.forward(conv2d64_out1);
        let relu6_out1 = burn::tensor::activation::relu(conv2d65_out1);
        let add13_out1 = relu6_out1.add(relu5_out1);
        let conv2d66_out1 = self.conv2d66.forward(add13_out1.clone());
        let conv2d67_out1 = self.conv2d67.forward(conv2d66_out1);
        let add14_out1 = add13_out1.add(conv2d67_out1);
        let relu7_out1 = burn::tensor::activation::relu(add14_out1);
        let conv2d68_out1 = self.conv2d68.forward(relu7_out1.clone());
        let conv2d69_out1 = self.conv2d69.forward(conv2d68_out1);
        let add15_out1 = relu7_out1.add(conv2d69_out1);
        let relu8_out1 = burn::tensor::activation::relu(add15_out1);
        let depthtospace3_out1 = {
            let [b, c, h, w] = relu8_out1.dims();
            let crd = relu8_out1
                .reshape([b, 2usize * 2usize, c / (2usize * 2usize), h, w])
                .swap_dims(1, 2)
                .reshape([b, c, h, w]);
            burn::nn::PixelShuffleConfig::new(2usize).init().forward(crd)
        };
        let pad3_out1 = depthtospace3_out1
            .pad(
                [(0usize, 0usize), (0usize, 0usize), (1usize, 1usize), (1usize, 1usize)],
                burn::tensor::ops::PadMode::Constant(0f32),
            );
        let conv2d70_out1 = self.conv2d70.forward(pad3_out1.clone());
        let conv2d71_out1 = self.conv2d71.forward(conv2d70_out1);
        let transpose5_out1 = conv2d71_out1.permute([0, 2, 3, 1]);
        let reshape5_out1 = transpose5_out1.reshape([1, -1, 12]);
        let concat1_out1 = burn::tensor::Tensor::cat(
            [reshape5_out1, reshape3_out1, reshape1_out1].into(),
            1,
        );
        let conv2d72_out1 = self.conv2d72.forward(pad3_out1);
        let conv2d73_out1 = self.conv2d73.forward(conv2d72_out1);
        let transpose6_out1 = conv2d73_out1.permute([0, 2, 3, 1]);
        let reshape6_out1 = transpose6_out1.reshape([1, -1, 1]);
        let concat2_out1 = burn::tensor::Tensor::cat(
            [reshape6_out1, reshape4_out1, reshape2_out1].into(),
            1,
        );
        (concat2_out1, concat1_out1)
    }
}
