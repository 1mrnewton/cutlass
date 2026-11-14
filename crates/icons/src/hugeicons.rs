//! Written by `tools/icongen` from `icons.txt`. Do not edit.
//!
//! Path data from `@hugeicons/core-free-icons` 4.4.0.
//!
//! MIT License
//!
//! Copyright (c) 2025 Hugeicons
//!
//! Permission is hereby granted, free of charge, to any person obtaining a copy
//! of this software and associated documentation files (the "Software"), to deal
//! in the Software without restriction, including without limitation the rights
//! to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//! copies of the Software, and to permit persons to whom the Software is
//! furnished to do so, subject to the following conditions:
//!
//! The above copyright notice and this permission notice shall be included in all
//! copies or substantial portions of the Software.
//!
//! THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//! IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//! FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//! AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//! LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//! OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//! SOFTWARE.

#[allow(unused_imports)]
use crate::{Cap, IconData, Join, Mark, Shape, Stroke};

pub static ICONS: &[IconData] = &[
    IconData {
        name: "media",
        source: Some("Image01Icon"),
        marks: &[
            Mark {
                shape: Shape::Circle {
                    cx: 7.5,
                    cy: 7.5,
                    r: 1.5,
                },
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
            Mark {
                shape: Shape::Path(
                    "M2.5 12C2.5 7.52166 2.5 5.28249 3.89124 3.89124C5.28249 2.5 7.52166 2.5 12 2.5C16.4783 2.5 18.7175 2.5 20.1088 3.89124C21.5 5.28249 21.5 7.52166 21.5 12C21.5 16.4783 21.5 18.7175 20.1088 20.1088C18.7175 21.5 16.4783 21.5 12 21.5C7.52166 21.5 5.28249 21.5 3.89124 20.1088C2.5 18.7175 2.5 16.4783 2.5 12Z",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Butt,
                    join: Join::Miter,
                }),
            },
            Mark {
                shape: Shape::Path("M5 21C9.37246 15.775 14.2741 8.88406 21.4975 13.5424"),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Butt,
                    join: Join::Miter,
                }),
            },
        ],
    },
    IconData {
        name: "audio",
        source: Some("AudioWave01Icon"),
        marks: &[
            Mark {
                shape: Shape::Path("M9 3V21"),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
            Mark {
                shape: Shape::Path("M6 7V17"),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
            Mark {
                shape: Shape::Path("M12 6V18"),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
            Mark {
                shape: Shape::Path("M15 9L15 15"),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
            Mark {
                shape: Shape::Path("M18 7L18 17"),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
            Mark {
                shape: Shape::Path("M21 11L21 13"),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
            Mark {
                shape: Shape::Path("M3 11L3 13"),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
        ],
    },
    IconData {
        name: "text",
        source: Some("TextIcon"),
        marks: &[
            Mark {
                shape: Shape::Path("M15 21.001H9"),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
            Mark {
                shape: Shape::Path(
                    "M12 3.00001V21.0008M12 3.00001C13.3874 3.00001 15.1695 3.03055 16.5884 3.17649C17.1885 3.2382 17.4886 3.26906 17.7541 3.37791C18.3066 3.60429 18.7518 4.10063 18.9194 4.67681C19 4.95382 19 5.26992 19 5.90215M12 3.00001C10.6126 3.00001 8.83047 3.03055 7.41161 3.17649C6.8115 3.2382 6.51144 3.26906 6.24586 3.37791C5.69344 3.60429 5.24816 4.10063 5.08057 4.67681C5 4.95382 5 5.26992 5 5.90215",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Miter,
                }),
            },
        ],
    },
    IconData {
        name: "stickers",
        source: Some("StickerIcon"),
        marks: &[
            Mark {
                shape: Shape::Path(
                    "M3 13V11C3 7.22876 3 5.34315 4.17157 4.17157C5.34315 3 7.22876 3 11 3H12.6863C14.3213 3 15.1388 3 15.8739 3.30449C16.609 3.60897 17.1871 4.18704 18.3432 5.34318L18.6569 5.65686C19.813 6.81298 20.391 7.39104 20.6955 8.12612C21 8.8612 21 9.67869 21 11.3137V13C21 16.7712 21 18.6569 19.8284 19.8284C18.6569 21 16.7712 21 13 21H11C7.22876 21 5.34315 21 4.17157 19.8284C3 18.6569 3 16.7712 3 13Z",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
            Mark {
                shape: Shape::Path(
                    "M15 3.5C15 5.84558 15 7.01836 15.6199 7.82628C15.7795 8.03428 15.9657 8.22046 16.1737 8.38006C16.9816 9 18.1544 9 20.5001 9",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
            Mark {
                shape: Shape::Path(
                    "M14.5 16C13.8152 16.6257 12.9459 17 12 17C11.0541 17 10.1848 16.6257 9.5 16M8.25 12.125V12.25M15.75 12.1257V12.2507M8.5 12.25C8.5 12.3881 8.38807 12.5 8.25 12.5C8.11193 12.5 8 12.3881 8 12.25C8 12.1119 8.11193 12 8.25 12C8.38807 12 8.5 12.1119 8.5 12.25ZM16 12.2507C16 12.3887 15.8881 12.5007 15.75 12.5007C15.6119 12.5007 15.5 12.3887 15.5 12.2507C15.5 12.1126 15.6119 12.0007 15.75 12.0007C15.8881 12.0007 16 12.1126 16 12.2507Z",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
        ],
    },
    IconData {
        name: "effects",
        source: Some("SparklesIcon"),
        marks: &[
            Mark {
                shape: Shape::Path(
                    "M15 2L15.5387 4.39157C15.9957 6.42015 17.5798 8.00431 19.6084 8.46127L22 9L19.6084 9.53873C17.5798 9.99569 15.9957 11.5798 15.5387 13.6084L15 16L14.4613 13.6084C14.0043 11.5798 12.4202 9.99569 10.3916 9.53873L8 9L10.3916 8.46127C12.4201 8.00431 14.0043 6.42015 14.4613 4.39158L15 2Z",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Butt,
                    join: Join::Round,
                }),
            },
            Mark {
                shape: Shape::Path(
                    "M7 12L7.38481 13.7083C7.71121 15.1572 8.84275 16.2888 10.2917 16.6152L12 17L10.2917 17.3848C8.84275 17.7112 7.71121 18.8427 7.38481 20.2917L7 22L6.61519 20.2917C6.28879 18.8427 5.15725 17.7112 3.70827 17.3848L2 17L3.70827 16.6152C5.15725 16.2888 6.28879 15.1573 6.61519 13.7083L7 12Z",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Butt,
                    join: Join::Round,
                }),
            },
        ],
    },
    IconData {
        name: "transitions",
        source: Some("TransitionLeftIcon"),
        marks: &[
            Mark {
                shape: Shape::Path(
                    "M22 6C22 4.59987 22 3.8998 21.7275 3.36502C21.4878 2.89462 21.1054 2.51217 20.635 2.27248C20.1002 2 19.4001 2 18 2C16.5999 2 15.8998 2 15.365 2.27248C14.8946 2.51217 14.5122 2.89462 14.2725 3.36502C14 3.8998 14 4.59987 14 6V18C14 19.4001 14 20.1002 14.2725 20.635C14.5122 21.1054 14.8946 21.4878 15.365 21.7275C15.8998 22 16.5999 22 18 22C19.4001 22 20.1002 22 20.635 21.7275C21.1054 21.4878 21.4878 21.1054 21.7275 20.635C22 20.1002 22 19.4001 22 18V6Z",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Butt,
                    join: Join::Miter,
                }),
            },
            Mark {
                shape: Shape::Path(
                    "M6 12H14M6 12C6 11.2998 7.9943 9.99153 8.5 9.5M6 12C6 12.7002 7.9943 14.0085 8.5 14.5",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
            Mark {
                shape: Shape::Path(
                    "M8 22C5.66111 22 4.49167 22 3.63789 21.4635C3.19267 21.1838 2.81621 20.8073 2.53647 20.3621C2 19.5083 2 18.3389 2 16V8C2 5.66111 2 4.49167 2.53647 3.63789C2.81621 3.19267 3.19267 2.81621 3.63789 2.53647C4.49167 2 5.66111 2 8 2",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Miter,
                }),
            },
        ],
    },
    IconData {
        name: "filters",
        source: Some("FilterIcon"),
        marks: &[Mark {
            shape: Shape::Path(
                "M8.85746 12.5061C6.36901 10.6456 4.59564 8.59915 3.62734 7.44867C3.3276 7.09253 3.22938 6.8319 3.17033 6.3728C2.96811 4.8008 2.86701 4.0148 3.32795 3.5074C3.7889 3 4.60404 3 6.23433 3H17.7657C19.396 3 20.2111 3 20.672 3.5074C21.133 4.0148 21.0319 4.8008 20.8297 6.37281C20.7706 6.83191 20.6724 7.09254 20.3726 7.44867C19.403 8.60062 17.6261 10.6507 15.1326 12.5135C14.907 12.6821 14.7583 12.9567 14.7307 13.2614C14.4837 15.992 14.2559 17.4876 14.1141 18.2442C13.8853 19.4657 12.1532 20.2006 11.226 20.8563C10.6741 21.2466 10.0043 20.782 9.93278 20.1778C9.79643 19.0261 9.53961 16.6864 9.25927 13.2614C9.23409 12.9539 9.08486 12.6761 8.85746 12.5061Z",
            ),
            fill: false,
            even_odd: false,
            stroke: Some(Stroke {
                width: 1.5,
                cap: Cap::Round,
                join: Join::Round,
            }),
        }],
    },
    IconData {
        name: "folder",
        source: Some("Folder01Icon"),
        marks: &[Mark {
            shape: Shape::Path(
                "M13.6656 6.49446H16C18.8284 6.49446 20.2426 6.49446 21.1213 7.37314C22 8.25182 22 9.66603 22 12.4945V14.5C22 17.3284 22 18.7426 21.1213 19.6213C20.2426 20.5 18.8284 20.5 16 20.5H8C5.17157 20.5 3.75736 20.5 2.87868 19.6213C2 18.7426 2 17.3284 2 14.5V8.99183C2 6.65409 2 5.48521 2.61604 4.67881C2.7778 4.46706 2.96706 4.2778 3.17881 4.11604C3.98521 3.5 5.15409 3.5 7.49183 3.5C8.16633 3.5 8.50358 3.5 8.81392 3.60442C8.8976 3.63257 8.97932 3.66623 9.05854 3.70518C9.35239 3.84965 9.59179 4.08719 10.0706 4.56227L10.8483 5.33389C11.4254 5.9066 11.714 6.19295 12.08 6.3437C12.446 6.49446 12.8525 6.49446 13.6656 6.49446Z",
            ),
            fill: false,
            even_odd: false,
            stroke: Some(Stroke {
                width: 1.5,
                cap: Cap::Round,
                join: Join::Round,
            }),
        }],
    },
    IconData {
        name: "adjustments",
        source: Some("SlidersHorizontalIcon"),
        marks: &[
            Mark {
                shape: Shape::Path("M2.99982 12.0002H10.9998"),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
            Mark {
                shape: Shape::Path("M13.9998 5.00015L20.9998 5.00015"),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
            Mark {
                shape: Shape::Path("M11.9998 19.0002L20.9998 19.0002"),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
            Mark {
                shape: Shape::Path(
                    "M2.99982 5.00015H9.99982M9.99982 5.00015V3.00015M9.99982 5.00015V7.00015",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
            Mark {
                shape: Shape::Path(
                    "M2.99982 19.0002H7.99982M7.99982 19.0002V17.0002M7.99982 19.0002V21.0002",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
            Mark {
                shape: Shape::Path(
                    "M14.9998 12.0002L20.9998 12.0002M14.9998 12.0002V10.0002M14.9998 12.0002V14.0002",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
        ],
    },
    IconData {
        name: "templates",
        source: Some("LayoutTemplateIcon"),
        marks: &[
            Mark {
                shape: Shape::Path(
                    "M17.7458 2.49963C19.3896 2.49963 20.2115 2.49963 20.7646 2.95361C20.8659 3.03672 20.9588 3.12958 21.0419 3.23085C21.4958 3.78402 21.4958 4.60589 21.4958 6.24963C21.4958 7.89338 21.4958 8.71525 21.0419 9.26842C20.9588 9.36969 20.8659 9.46255 20.7646 9.54566C20.2115 9.99963 19.3896 9.99963 17.7458 9.99963L6.74585 9.99963C5.10211 9.99963 4.28024 9.99963 3.72706 9.54565C3.62579 9.46255 3.53294 9.36969 3.44983 9.26842C2.99585 8.71525 2.99585 7.89337 2.99585 6.24963C2.99585 4.60589 2.99585 3.78402 3.44983 3.23085C3.53294 3.12958 3.62579 3.03672 3.72706 2.95361C4.28024 2.49963 5.10211 2.49963 6.74585 2.49963L17.7458 2.49963Z",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Butt,
                    join: Join::Round,
                }),
            },
            Mark {
                shape: Shape::Path(
                    "M17.7459 13.9998C19.3898 13.9997 20.2117 13.9997 20.7649 14.4537C20.8661 14.5368 20.959 14.6296 21.0421 14.7308C21.4961 15.284 21.4961 16.106 21.4961 17.7499C21.4961 19.3936 21.4961 20.2155 21.0421 20.7687C20.959 20.87 20.8662 20.9628 20.765 21.0459C20.2118 21.4999 19.3899 21.4999 17.7461 21.4999L15.7458 21.5C14.1019 21.5 13.28 21.5001 12.7268 21.046C12.6256 20.963 12.5327 20.8701 12.4496 20.7689C11.9956 20.2157 11.9956 19.3938 11.9956 17.7499C11.9956 16.1061 11.9956 15.2842 12.4496 14.731C12.5327 14.6298 12.6255 14.537 12.7267 14.4539C13.2799 13.9999 14.1018 13.9999 15.7456 13.9998L17.7459 13.9998Z",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Butt,
                    join: Join::Round,
                }),
            },
            Mark {
                shape: Shape::Path(
                    "M5.24559 13.9999C5.94314 13.9999 6.29192 13.9999 6.57574 14.0859C7.21509 14.2798 7.71542 14.7801 7.9093 15.4195C7.99536 15.7033 7.99536 16.0521 7.99536 16.7496V18.7505C7.99536 19.4481 7.99536 19.7968 7.9093 20.0806C7.71542 20.72 7.2151 21.2203 6.57576 21.4142C6.29195 21.5003 5.94317 21.5003 5.24563 21.5003C4.54807 21.5003 4.19929 21.5003 3.91548 21.4142C3.27612 21.2203 2.77579 20.72 2.58191 20.0807C2.49585 19.7968 2.49585 19.4481 2.49585 18.7505L2.49585 16.7496C2.49585 16.0521 2.49585 15.7033 2.58191 15.4195C2.77579 14.7802 3.27611 14.2798 3.91545 14.086C4.19926 13.9999 4.54804 13.9999 5.24559 13.9999Z",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Butt,
                    join: Join::Round,
                }),
            },
        ],
    },
    IconData {
        name: "video",
        source: Some("Video01Icon"),
        marks: &[
            Mark {
                shape: Shape::Path(
                    "M2 11C2 7.70017 2 6.05025 3.02513 5.02513C4.05025 4 5.70017 4 9 4H10C13.2998 4 14.9497 4 15.9749 5.02513C17 6.05025 17 7.70017 17 11V13C17 16.2998 17 17.9497 15.9749 18.9749C14.9497 20 13.2998 20 10 20H9C5.70017 20 4.05025 20 3.02513 18.9749C2 17.9497 2 16.2998 2 13V11Z",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Butt,
                    join: Join::Miter,
                }),
            },
            Mark {
                shape: Shape::Path(
                    "M17 8.90585L17.1259 8.80196C19.2417 7.05623 20.2996 6.18336 21.1498 6.60482C22 7.02628 22 8.42355 22 11.2181V12.7819C22 15.5765 22 16.9737 21.1498 17.3952C20.2996 17.8166 19.2417 16.9438 17.1259 15.198L17 15.0941",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Miter,
                }),
            },
            Mark {
                shape: Shape::Circle {
                    cx: 11.5,
                    cy: 9.5,
                    r: 1.5,
                },
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Butt,
                    join: Join::Miter,
                }),
            },
        ],
    },
    IconData {
        name: "textVideo",
        source: Some("ImageToVideoIcon"),
        marks: &[
            Mark {
                shape: Shape::Path(
                    "M2 13L5.46967 9.53033C5.80923 9.19077 6.26978 9 6.75 9C7.23022 9 7.69077 9.19077 8.03033 9.53033L9 10.5",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
            Mark {
                shape: Shape::Path(
                    "M9 18C6.19108 18 4.78661 18 3.77772 17.3259C3.34096 17.034 2.96596 16.659 2.67412 16.2223C2 15.2134 2 13.8089 2 11V10C2 6.22876 2 4.34315 3.17157 3.17157C4.34315 2 6.22876 2 10 2H11C13.8089 2 15.2134 2 16.2223 2.67412C16.659 2.96596 17.034 3.34096 17.3259 3.77772C18 4.78661 18 6.19108 18 9",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
            Mark {
                shape: Shape::Path(
                    "M21.5981 13.5C22 14.1962 22 15.1308 22 17C22 18.8692 22 19.8038 21.5981 20.5C21.3348 20.9561 20.9561 21.3348 20.5 21.5981C19.8038 22 18.8692 22 17 22C15.1308 22 14.1962 22 13.5 21.5981C13.0439 21.3348 12.6652 20.9561 12.4019 20.5C12 19.8038 12 18.8692 12 17C12 15.1308 12 14.1962 12.4019 13.5C12.6652 13.0439 13.0439 12.6652 13.5 12.4019C14.1962 12 15.1308 12 17 12C18.8692 12 19.8038 12 20.5 12.4019C20.9561 12.6652 21.3348 13.0439 21.5981 13.5Z",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
            Mark {
                shape: Shape::Path(
                    "M16.5638 17H17.6915M16 15.8547V18.1453C16 18.4206 16.2869 18.5906 16.5124 18.449L18.3357 17.3037C18.5548 17.1661 18.5548 16.8339 18.3357 16.6963L16.5124 15.551C16.2869 15.4094 16 15.5794 16 15.8547Z",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
            Mark {
                shape: Shape::Path(
                    "M13.5 7C13.7761 7 14 6.77614 14 6.5C14 6.22386 13.7761 6 13.5 6M13.5 7C13.2239 7 13 6.77614 13 6.5C13 6.22386 13.2239 6 13.5 6M13.5 7V6",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
        ],
    },
    IconData {
        name: "composition",
        source: Some("LayerIcon"),
        marks: &[
            Mark {
                shape: Shape::Path(
                    "M9.60573 4.81298C10.7856 4.27099 11.3755 4 12 4C12.6245 4 13.2144 4.27099 14.3943 4.81298L19.2873 7.06064C21.0958 7.89137 22 8.30674 22 9C22 9.69326 21.0958 10.1086 19.2873 10.9394L14.3943 13.187C13.2144 13.729 12.6245 14 12 14C11.3755 14 10.7856 13.729 9.60573 13.187L4.7127 10.9394C2.90423 10.1086 2 9.69326 2 9C2 8.30674 2.90423 7.89137 4.7127 7.06064L9.60573 4.81298Z",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
            Mark {
                shape: Shape::Path(
                    "M20.2327 13.5C21.4109 14.062 22 14.4405 22 15.0001C22 15.6934 21.0958 16.1087 19.2873 16.9395L14.3943 19.1871C13.2144 19.7291 12.6245 20.0001 12 20.0001C11.3755 20.0001 10.7856 19.7291 9.60573 19.1871L4.7127 16.9395C2.90423 16.1087 2 15.6934 2 15.0001C2 14.4405 2.58909 14.062 3.76727 13.5",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
        ],
    },
    IconData {
        name: "plugin",
        source: Some("MagicWand01Icon"),
        marks: &[
            Mark {
                shape: Shape::Path(
                    "M13.9258 12.7775L11.7775 10.6292C11.4847 10.3364 11.3383 10.19 11.1803 10.1117C10.8798 9.96277 10.527 9.96277 10.2264 10.1117C10.0685 10.19 9.92207 10.3364 9.62923 10.6292C9.33638 10.9221 9.18996 11.0685 9.11169 11.2264C8.96277 11.527 8.96277 11.8798 9.11169 12.1803C9.18996 12.3383 9.33638 12.4847 9.62923 12.7775L11.7775 14.9258M13.9258 12.7775L20.3708 19.2225C20.6636 19.5153 20.81 19.6617 20.8883 19.8197C21.0372 20.1202 21.0372 20.473 20.8883 20.7736C20.81 20.9315 20.6636 21.0779 20.3708 21.3708C20.0779 21.6636 19.9315 21.81 19.7736 21.8883C19.473 22.0372 19.1202 22.0372 18.8197 21.8883C18.6617 21.81 18.5153 21.6636 18.2225 21.3708L11.7775 14.9258M13.9258 12.7775L11.7775 14.9258",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
            Mark {
                shape: Shape::Path(
                    "M17 2L17.2948 2.7966C17.6813 3.84117 17.8746 4.36345 18.2556 4.74445C18.6366 5.12545 19.1588 5.31871 20.2034 5.70523L21 6L20.2034 6.29477C19.1588 6.68129 18.6366 6.87456 18.2556 7.25555C17.8746 7.63655 17.6813 8.15883 17.2948 9.2034L17 10L16.7052 9.2034C16.3187 8.15884 16.1254 7.63655 15.7444 7.25555C15.3634 6.87455 14.8412 6.68129 13.7966 6.29477L13 6L13.7966 5.70523C14.8412 5.31871 15.3634 5.12545 15.7444 4.74445C16.1254 4.36345 16.3187 3.84117 16.7052 2.7966L17 2Z",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Butt,
                    join: Join::Round,
                }),
            },
            Mark {
                shape: Shape::Path(
                    "M6 4L6.22108 4.59745C6.51097 5.38087 6.65592 5.77259 6.94167 6.05834C7.22741 6.34408 7.61913 6.48903 8.40255 6.77892L9 7L8.40255 7.22108C7.61913 7.51097 7.22741 7.65592 6.94166 7.94167C6.65592 8.22741 6.51097 8.61913 6.22108 9.40255L6 10L5.77892 9.40255C5.48903 8.61913 5.34408 8.22741 5.05833 7.94167C4.77259 7.65592 4.38087 7.51097 3.59745 7.22108L3 7L3.59745 6.77892C4.38087 6.48903 4.77259 6.34408 5.05833 6.05833C5.34408 5.77259 5.48903 5.38087 5.77892 4.59745L6 4Z",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Butt,
                    join: Join::Round,
                }),
            },
        ],
    },
    IconData {
        name: "ad",
        source: Some("CubeIcon"),
        marks: &[Mark {
            shape: Shape::Path(
                "M2.79289 21.2071C3.08579 21.5 3.55719 21.5 4.5 21.5H14.5C15.4428 21.5 15.9142 21.5 16.2071 21.2071M2.79289 21.2071C2.5 20.9142 2.5 20.4428 2.5 19.5V9.5C2.5 8.55719 2.5 8.08579 2.79289 7.79289M2.79289 21.2071L8.79289 15.2071M16.2071 21.2071C16.5 20.9142 16.5 20.4428 16.5 19.5V9.5C16.5 8.55719 16.5 8.08579 16.2071 7.79289M16.2071 21.2071L21.2071 16.2071C21.5 15.9142 21.5 15.4428 21.5 14.5V4.5C21.5 3.55719 21.5 3.08579 21.2071 2.79289M16.2071 7.79289C15.9142 7.5 15.4428 7.5 14.5 7.5H4.5C3.55719 7.5 3.08579 7.5 2.79289 7.79289M16.2071 7.79289L21.2071 2.79289M2.79289 7.79289L7.79289 2.79289C8.08579 2.5 8.55719 2.5 9.5 2.5H19.5C20.4428 2.5 20.9142 2.5 21.2071 2.79289M8.79289 15.2071C9.08579 15.5 9.55719 15.5 10.5 15.5H14M8.79289 15.2071C8.5 14.9142 8.5 14.4428 8.5 13.5V10.5",
            ),
            fill: false,
            even_odd: false,
            stroke: Some(Stroke {
                width: 1.5,
                cap: Cap::Round,
                join: Join::Round,
            }),
        }],
    },
    IconData {
        name: "mixed",
        source: Some("BlendIcon"),
        marks: &[
            Mark {
                shape: Shape::Circle {
                    cx: 17.0,
                    cy: 17.0,
                    r: 5.0,
                },
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Butt,
                    join: Join::Miter,
                }),
            },
            Mark {
                shape: Shape::Path(
                    "M7.5 13H6.4C4.32582 13 3.28873 13 2.64437 12.3556C2 11.7113 2 10.6742 2 8.6V6.4C2 4.32582 2 3.28873 2.64437 2.64437C3.28873 2 4.32582 2 6.4 2H8.6C10.6742 2 11.7113 2 12.3556 2.64437C13 3.28873 13 4.32582 13 6.4V7.5",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Butt,
                    join: Join::Miter,
                }),
            },
            Mark {
                shape: Shape::Path(
                    "M12.0348 19C9.17734 18.5206 7 16.0355 7 13.0418C7 9.70499 9.70499 7 13.0418 7C16.0355 7 18.5206 9.17734 19 12.0348",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Butt,
                    join: Join::Miter,
                }),
            },
        ],
    },
    IconData {
        name: "play",
        source: Some("PlayIcon"),
        marks: &[Mark {
            shape: Shape::Path(
                "M18.8906 12.846C18.5371 14.189 16.8667 15.138 13.5257 17.0361C10.296 18.8709 8.6812 19.7884 7.37983 19.4196C6.8418 19.2671 6.35159 18.9776 5.95624 18.5787C5 17.6139 5 15.7426 5 12C5 8.2574 5 6.3861 5.95624 5.42132C6.35159 5.02245 6.8418 4.73288 7.37983 4.58042C8.6812 4.21165 10.296 5.12907 13.5257 6.96393C16.8667 8.86197 18.5371 9.811 18.8906 11.154C19.0365 11.7084 19.0365 12.2916 18.8906 12.846Z",
            ),
            fill: false,
            even_odd: false,
            stroke: Some(Stroke {
                width: 1.5,
                cap: Cap::Butt,
                join: Join::Round,
            }),
        }],
    },
    IconData {
        name: "plus",
        source: Some("PlusSignIcon"),
        marks: &[Mark {
            shape: Shape::Path("M12 4V20M20 12H4"),
            fill: false,
            even_odd: false,
            stroke: Some(Stroke {
                width: 1.5,
                cap: Cap::Round,
                join: Join::Round,
            }),
        }],
    },
    IconData {
        name: "pause",
        source: Some("PauseIcon"),
        marks: &[
            Mark {
                shape: Shape::Path(
                    "M4 7C4 5.58579 4 4.87868 4.43934 4.43934C4.87868 4 5.58579 4 7 4C8.41421 4 9.12132 4 9.56066 4.43934C10 4.87868 10 5.58579 10 7V17C10 18.4142 10 19.1213 9.56066 19.5607C9.12132 20 8.41421 20 7 20C5.58579 20 4.87868 20 4.43934 19.5607C4 19.1213 4 18.4142 4 17V7Z",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Butt,
                    join: Join::Miter,
                }),
            },
            Mark {
                shape: Shape::Path(
                    "M14 7C14 5.58579 14 4.87868 14.4393 4.43934C14.8787 4 15.5858 4 17 4C18.4142 4 19.1213 4 19.5607 4.43934C20 4.87868 20 5.58579 20 7V17C20 18.4142 20 19.1213 19.5607 19.5607C19.1213 20 18.4142 20 17 20C15.5858 20 14.8787 20 14.4393 19.5607C14 19.1213 14 18.4142 14 17V7Z",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Butt,
                    join: Join::Miter,
                }),
            },
        ],
    },
    IconData {
        name: "camera",
        source: Some("Camera01Icon"),
        marks: &[
            Mark {
                shape: Shape::Path(
                    "M12.6974 3.5H11.303C10.5884 3.5 10.2311 3.5 9.91067 3.612C9.71499 3.68039 9.53113 3.77879 9.36568 3.90367C9.09474 4.10816 8.89655 4.40544 8.50018 5L8.50017 5.00001C8.29717 5.30453 7.99794 5.75337 7.87867 5.87871C7.58314 6.18927 7.19563 6.39666 6.77329 6.47029C6.60284 6.5 6.41985 6.5 6.05387 6.5C5.07379 6.5 4.58376 6.5 4.18307 6.61342C3.18074 6.89716 2.39734 7.68055 2.1136 8.68289C2.00018 9.08357 2.00018 9.57361 2.00018 10.5537V14.5C2.00018 17.3284 2.00018 18.7426 2.87886 19.6213C3.75754 20.5 5.17176 20.5 8.00018 20.5H16.0002C18.8286 20.5 20.2428 20.5 21.1215 19.6213C22.0002 18.7426 22.0002 17.3284 22.0002 14.5V10.5537C22.0002 9.57361 22.0002 9.08357 21.8868 8.68289C21.603 7.68055 20.8196 6.89716 19.8173 6.61342C19.4166 6.5 18.9266 6.5 17.9465 6.5C17.5805 6.5 17.3975 6.5 17.2271 6.47029C16.8047 6.39666 16.4172 6.18927 16.1217 5.87871C16.0024 5.75336 15.7032 5.30451 15.5002 5C15.1038 4.40544 14.9056 4.10816 14.6347 3.90367C14.4692 3.77879 14.2854 3.68039 14.0897 3.612C13.7693 3.5 13.412 3.5 12.6974 3.5Z",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
            Mark {
                shape: Shape::Path(
                    "M16.0002 13C16.0002 15.2091 14.2093 17 12.0002 17C9.79104 17 8.00018 15.2091 8.00018 13C8.00018 10.7909 9.79104 9 12.0002 9C14.2093 9 16.0002 10.7909 16.0002 13Z",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
            Mark {
                shape: Shape::Path(
                    "M19.1252 9.5H19.0002M19.2502 9.5C19.2502 9.63807 19.1383 9.75 19.0002 9.75C18.8621 9.75 18.7502 9.63807 18.7502 9.5C18.7502 9.36193 18.8621 9.25 19.0002 9.25C19.1383 9.25 19.2502 9.36193 19.2502 9.5Z",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Miter,
                }),
            },
        ],
    },
    IconData {
        name: "chevron",
        source: Some("ArrowRight01Icon"),
        marks: &[Mark {
            shape: Shape::Path("M9.00005 6C9.00005 6 15 10.4189 15 12C15 13.5812 9 18 9 18"),
            fill: false,
            even_odd: false,
            stroke: Some(Stroke {
                width: 1.5,
                cap: Cap::Round,
                join: Join::Round,
            }),
        }],
    },
    IconData {
        name: "chevronLeft",
        source: Some("ArrowLeft01Icon"),
        marks: &[Mark {
            shape: Shape::Path("M15 6C15 6 9.00001 10.4189 9 12C8.99999 13.5812 15 18 15 18"),
            fill: false,
            even_odd: false,
            stroke: Some(Stroke {
                width: 1.5,
                cap: Cap::Round,
                join: Join::Round,
            }),
        }],
    },
    IconData {
        name: "alignLeft",
        source: Some("AlignLeftIcon"),
        marks: &[
            Mark {
                shape: Shape::Path(
                    "M8.00232 7.5C8.00232 6.65611 7.91895 5.68211 8.75232 5.20096C9.10039 5 9.5677 5 10.5023 5H11.5023C12.4369 5 12.9042 5 13.2523 5.20096C14.0857 5.68211 14.0023 6.65611 14.0023 7.5C14.0023 8.34389 14.0857 9.31789 13.2523 9.79904C12.9042 10 12.4369 10 11.5023 10H10.5023C9.5677 10 9.10039 10 8.75232 9.79904C7.91895 9.31789 8.00232 8.34389 8.00232 7.5Z",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
            Mark {
                shape: Shape::Path(
                    "M8.00232 16.5C8.00232 15.6561 7.91895 14.6821 8.75232 14.201C9.10039 14 9.5677 14 10.5023 14H17.5023C18.4369 14 18.9042 14 19.2523 14.201C20.0857 14.6821 20.0023 15.6561 20.0023 16.5C20.0023 17.3439 20.0857 18.3179 19.2523 18.799C18.9042 19 18.4369 19 17.5023 19H10.5023C9.5677 19 9.10039 19 8.75232 18.799C7.91895 18.3179 8.00232 17.3439 8.00232 16.5Z",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
            Mark {
                shape: Shape::Path("M4 2V22"),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
        ],
    },
    IconData {
        name: "alignCenter",
        source: Some("AlignHorizontalCenterIcon"),
        marks: &[
            Mark {
                shape: Shape::Path("M12 19L12 22"),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
            Mark {
                shape: Shape::Path("M12 10L12 14"),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
            Mark {
                shape: Shape::Path("M12 2L12 5"),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
            Mark {
                shape: Shape::Path(
                    "M8.00036 7.5C8.00036 6.65611 7.91699 5.68211 8.75036 5.20096C9.09844 5 9.56575 5 10.5004 5H13.5004C14.435 5 14.9023 5 15.2504 5.20096C16.0837 5.68211 16.0004 6.65611 16.0004 7.5C16.0004 8.34389 16.0837 9.31789 15.2504 9.79904C14.9023 10 14.435 10 13.5004 10H10.5004C9.56575 10 9.09844 10 8.75036 9.79904C7.91699 9.31789 8.00036 8.34389 8.00036 7.5Z",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
            Mark {
                shape: Shape::Path(
                    "M4.00036 16.5C4.00036 15.6561 3.91699 14.6821 4.75036 14.201C5.09844 14 5.56575 14 6.50036 14H17.5004C18.435 14 18.9023 14 19.2504 14.201C20.0837 14.6821 20.0004 15.6561 20.0004 16.5C20.0004 17.3439 20.0837 18.3179 19.2504 18.799C18.9023 19 18.435 19 17.5004 19H6.50036C5.56575 19 5.09844 19 4.75036 18.799C3.91699 18.3179 4.00036 17.3439 4.00036 16.5Z",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
        ],
    },
    IconData {
        name: "alignRight",
        source: Some("AlignRightIcon"),
        marks: &[
            Mark {
                shape: Shape::Path(
                    "M10.0004 7.5C10.0004 6.65611 9.91699 5.68211 10.7504 5.20096C11.0984 5 11.5657 5 12.5004 5H13.5004C14.435 5 14.9023 5 15.2504 5.20096C16.0837 5.68211 16.0004 6.65611 16.0004 7.5C16.0004 8.34389 16.0837 9.31789 15.2504 9.79904C14.9023 10 14.435 10 13.5004 10H12.5004C11.5657 10 11.0984 10 10.7504 9.79904C9.91699 9.31789 10.0004 8.34389 10.0004 7.5Z",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
            Mark {
                shape: Shape::Path(
                    "M4.00036 16.5C4.00036 15.6561 3.91699 14.6821 4.75036 14.201C5.09844 14 5.56575 14 6.50036 14H13.5004C14.435 14 14.9023 14 15.2504 14.201C16.0837 14.6821 16.0004 15.6561 16.0004 16.5C16.0004 17.3439 16.0837 18.3179 15.2504 18.799C14.9023 19 14.435 19 13.5004 19H6.50036C5.56575 19 5.09844 19 4.75036 18.799C3.91699 18.3179 4.00036 17.3439 4.00036 16.5Z",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
            Mark {
                shape: Shape::Path("M20 2V22"),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
        ],
    },
    IconData {
        name: "alignTop",
        source: Some("AlignTopIcon"),
        marks: &[
            Mark {
                shape: Shape::Path(
                    "M16.502 7.99988C17.3458 7.99988 18.3198 7.9165 18.801 8.74988C19.002 9.09795 19.002 9.56526 19.002 10.4999V11.4999C19.002 12.4345 19.002 12.9018 18.801 13.2499C18.3198 14.0832 17.3458 13.9999 16.502 13.9999C15.6581 13.9999 14.6841 14.0832 14.2029 13.2499C14.002 12.9018 14.002 12.4345 14.002 11.4999L14.002 10.4999C14.002 9.56526 14.002 9.09795 14.2029 8.74988C14.6841 7.9165 15.6581 7.99988 16.502 7.99988Z",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
            Mark {
                shape: Shape::Path(
                    "M7.50195 7.99988C8.34585 7.99988 9.31984 7.9165 9.80099 8.74988C10.002 9.09795 10.002 9.56526 10.002 10.4999L10.002 17.4999C10.002 18.4345 10.002 18.9018 9.80099 19.2499C9.31984 20.0832 8.34585 19.9999 7.50195 19.9999C6.65806 19.9999 5.68406 20.0832 5.20292 19.2499C5.00195 18.9018 5.00195 18.4345 5.00195 17.4999L5.00195 10.4999C5.00195 9.56526 5.00195 9.09795 5.20292 8.74988C5.68406 7.9165 6.65806 7.99988 7.50195 7.99988Z",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
            Mark {
                shape: Shape::Path("M22 4L2 4"),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
        ],
    },
    IconData {
        name: "alignMiddle",
        source: Some("AlignVerticalCenterIcon"),
        marks: &[
            Mark {
                shape: Shape::Path(
                    "M16.5 8.00232C17.3439 8.00232 18.3179 7.91895 18.799 8.75232C19 9.10039 19 9.5677 19 10.5023V13.5023C19 14.4369 19 14.9042 18.799 15.2523C18.3179 16.0857 17.3439 16.0023 16.5 16.0023C15.6561 16.0023 14.6821 16.0857 14.201 15.2523C14 14.9042 14 14.4369 14 13.5023L14 10.5023C14 9.5677 14 9.10039 14.201 8.75232C14.6821 7.91895 15.6561 8.00232 16.5 8.00232Z",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
            Mark {
                shape: Shape::Path(
                    "M7.5 4.00232C8.34389 4.00232 9.31789 3.91895 9.79904 4.75232C10 5.10039 10 5.5677 10 6.50232L10 17.5023C10 18.4369 10 18.9042 9.79904 19.2523C9.31789 20.0857 8.34389 20.0023 7.5 20.0023C6.65611 20.0023 5.68211 20.0857 5.20096 19.2523C5 18.9042 5 18.4369 5 17.5023L5 6.50232C5 5.5677 5 5.10039 5.20096 4.75232C5.68211 3.91895 6.65611 4.00232 7.5 4.00232Z",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
            Mark {
                shape: Shape::Path("M5 12H2"),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
            Mark {
                shape: Shape::Path("M14 12L10 12"),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
            Mark {
                shape: Shape::Path("M22 12L19 12"),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
        ],
    },
    IconData {
        name: "alignBottom",
        source: Some("AlignBottomIcon"),
        marks: &[
            Mark {
                shape: Shape::Path(
                    "M16.502 10.0023C17.3458 10.0023 18.3198 9.91895 18.801 10.7523C19.002 11.1004 19.002 11.5677 19.002 12.5023V13.5023C19.002 14.4369 19.002 14.9042 18.801 15.2523C18.3198 16.0857 17.3458 16.0023 16.502 16.0023C15.6581 16.0023 14.6841 16.0857 14.2029 15.2523C14.002 14.9042 14.002 14.4369 14.002 13.5023L14.002 12.5023C14.002 11.5677 14.002 11.1004 14.2029 10.7523C14.6841 9.91895 15.6581 10.0023 16.502 10.0023Z",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
            Mark {
                shape: Shape::Path(
                    "M7.50195 4.00232C8.34585 4.00232 9.31984 3.91895 9.80099 4.75232C10.002 5.10039 10.002 5.5677 10.002 6.50232L10.002 13.5023C10.002 14.4369 10.002 14.9042 9.80099 15.2523C9.31984 16.0857 8.34585 16.0023 7.50195 16.0023C6.65806 16.0023 5.68406 16.0857 5.20292 15.2523C5.00195 14.9042 5.00195 14.4369 5.00195 13.5023L5.00195 6.50232C5.00195 5.5677 5.00195 5.10039 5.20292 4.75232C5.68406 3.91895 6.65806 4.00232 7.50195 4.00232Z",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
            Mark {
                shape: Shape::Path("M22 20L2 20"),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
        ],
    },
    IconData {
        name: "keyframe",
        source: Some("KeyframeIcon"),
        marks: &[Mark {
            shape: Shape::Path(
                "M9.38188 5.00507C10.5901 3.66836 11.1943 3 12 3C12.8057 3 13.4099 3.66836 14.6181 5.00507L17.5897 8.29254C19.1966 10.0702 20 10.9591 20 12C20 13.0409 19.1966 13.9298 17.5897 15.7075L14.6181 18.9949C13.4099 20.3316 12.8057 21 12 21C11.1943 21 10.5901 20.3316 9.38188 18.9949L6.41031 15.7075C4.80344 13.9298 4 13.0409 4 12C4 10.9591 4.80344 10.0702 6.41031 8.29254L9.38188 5.00507Z",
            ),
            fill: false,
            even_odd: false,
            stroke: Some(Stroke {
                width: 1.5,
                cap: Cap::Butt,
                join: Join::Miter,
            }),
        }],
    },
    IconData {
        name: "graphs",
        source: Some("EaseInOutIcon"),
        marks: &[Mark {
            shape: Shape::Path("M21 4C11.0535 4 12.9465 20 3 20"),
            fill: false,
            even_odd: false,
            stroke: Some(Stroke {
                width: 1.5,
                cap: Cap::Round,
                join: Join::Round,
            }),
        }],
    },
    IconData {
        name: "reset",
        source: Some("RefreshIcon"),
        marks: &[Mark {
            shape: Shape::Path(
                "M20.0092 2V5.13219C20.0092 5.42605 19.6418 5.55908 19.4537 5.33333C17.6226 3.2875 14.9617 2 12 2C6.47715 2 2 6.47715 2 12C2 17.5228 6.47715 22 12 22C17.5228 22 22 17.5228 22 12",
            ),
            fill: false,
            even_odd: false,
            stroke: Some(Stroke {
                width: 1.5,
                cap: Cap::Round,
                join: Join::Round,
            }),
        }],
    },
    IconData {
        name: "rotate",
        source: Some("RotateClockwiseIcon"),
        marks: &[
            Mark {
                shape: Shape::Path(
                    "M4 2V5.13219C4 5.42605 4.36724 5.55908 4.55527 5.33333C6.3854 3.2875 9.04499 2 12.0051 2C17.5251 2 22 6.47715 22 12C22 15.9582 19.7015 19.3793 16.367 21",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
            Mark {
                shape: Shape::Path(
                    "M11.7347 22.0001C12.2016 22.0001 12.6611 21.9688 13.1111 21.9084M2.26537 8.66675C2.15297 9.06394 2.06477 9.46536 2 9.86901M2.03457 13.5381C2.10487 13.9381 2.19644 14.3343 2.30852 14.7245M3.83292 17.9963C4.07124 18.3497 4.3296 18.69 4.6071 19.0147M7.42857 21.3607C7.78228 21.5632 8.15042 21.7464 8.53228 21.9084",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
        ],
    },
    IconData {
        name: "search",
        source: Some("Search01Icon"),
        marks: &[
            Mark {
                shape: Shape::Path("M17 17L21 21"),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
            Mark {
                shape: Shape::Path(
                    "M19 11C19 6.58172 15.4183 3 11 3C6.58172 3 3 6.58172 3 11C3 15.4183 6.58172 19 11 19C15.4183 19 19 15.4183 19 11Z",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
        ],
    },
    IconData {
        name: "star",
        source: Some("StarIcon"),
        marks: &[Mark {
            shape: Shape::Path(
                "M12.7972 2.90494C13.2554 3.57043 13.9479 4.70769 14.6514 6.04507C15.1942 7.07701 15.4657 7.59298 15.8551 7.86959C16.2444 8.14621 16.812 8.2313 17.9472 8.40149C19.5204 8.63736 20.6806 8.89214 21.3467 9.05516C21.735 9.15017 21.9976 9.50036 21.9976 9.9001C21.9976 10.1262 21.9127 10.344 21.7606 10.5113C21.0637 11.2777 20.3487 12.0836 19.4701 12.961C18.605 13.8249 18.1724 14.2569 18.0253 14.7261C17.8781 15.1954 17.9834 15.7792 18.1939 16.9469C18.3735 17.9433 18.5569 19.1306 18.6921 20.4219C18.7528 21.0024 18.2921 21.5 17.7085 21.5C17.6522 21.5 17.5967 21.4962 17.542 21.4824C17.0502 21.3578 15.919 20.7845 14.1246 19.7243C13.0722 19.1025 12.5461 18.7917 11.9993 18.7917C11.4526 18.7917 10.9265 19.1026 9.87427 19.7244C8.08025 20.7845 6.94977 21.3578 6.45797 21.4824C6.40334 21.4962 6.34784 21.5 6.29148 21.5C5.70788 21.5 5.24716 21.0024 5.30793 20.4219C5.44314 19.1306 5.6265 17.9433 5.80612 16.9469C6.01661 15.7792 6.12186 15.1954 5.97471 14.7261C5.82756 14.2569 5.395 13.8249 4.52987 12.961C3.65132 12.0836 2.93633 11.2777 2.23944 10.5113C2.08729 10.344 2.00244 10.1262 2.00244 9.9001C2.00244 9.50036 2.265 9.15017 2.65328 9.05516C3.31945 8.89214 4.4796 8.63736 6.05284 8.40149C7.18801 8.2313 7.75559 8.14621 8.14498 7.86958C8.53437 7.59295 8.80577 7.07698 9.34859 6.04505C10.0521 4.70768 10.7445 3.57044 11.2026 2.90495C11.3825 2.64364 11.6827 2.5 11.9999 2.5C12.3171 2.5 12.6173 2.64364 12.7972 2.90494Z",
            ),
            fill: false,
            even_odd: false,
            stroke: Some(Stroke {
                width: 1.5,
                cap: Cap::Round,
                join: Join::Round,
            }),
        }],
    },
    IconData {
        name: "bold",
        source: Some("TextBoldIcon"),
        marks: &[
            Mark {
                shape: Shape::Path(
                    "M5 6C5 4.58579 5 3.87868 5.43934 3.43934C5.87868 3 6.58579 3 8 3H12.5789C15.0206 3 17 5.01472 17 7.5C17 9.98528 15.0206 12 12.5789 12H5V6Z",
                ),
                fill: false,
                even_odd: true,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
            Mark {
                shape: Shape::Path(
                    "M12.4286 12H13.6667C16.0599 12 18 14.0147 18 16.5C18 18.9853 16.0599 21 13.6667 21H8C6.58579 21 5.87868 21 5.43934 20.5607C5 20.1213 5 19.4142 5 18V12",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
        ],
    },
    IconData {
        name: "italic",
        source: Some("TextItalicIcon"),
        marks: &[
            Mark {
                shape: Shape::Path("M12 4H19"),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Miter,
                }),
            },
            Mark {
                shape: Shape::Path("M8 20L16 4"),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Miter,
                }),
            },
            Mark {
                shape: Shape::Path("M5 20H12"),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Miter,
                }),
            },
        ],
    },
    IconData {
        name: "underline",
        source: Some("TextUnderlineIcon"),
        marks: &[
            Mark {
                shape: Shape::Path(
                    "M5.5 3V11.5C5.5 15.0899 8.41015 18 12 18C15.5899 18 18.5 15.0899 18.5 11.5V3",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
            Mark {
                shape: Shape::Path("M3 21H21"),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Miter,
                }),
            },
        ],
    },
    IconData {
        name: "textLeft",
        source: Some("TextAlignLeftIcon"),
        marks: &[
            Mark {
                shape: Shape::Path("M3 3H21"),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
            Mark {
                shape: Shape::Path("M3 9H11"),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
            Mark {
                shape: Shape::Path("M3 15H21"),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
            Mark {
                shape: Shape::Path("M3 21H11"),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
        ],
    },
    IconData {
        name: "textCenter",
        source: Some("TextAlignCenterIcon"),
        marks: &[
            Mark {
                shape: Shape::Path("M3 3H21"),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
            Mark {
                shape: Shape::Path("M8 9H16"),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
            Mark {
                shape: Shape::Path("M3 15H21"),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
            Mark {
                shape: Shape::Path("M8 21H16"),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
        ],
    },
    IconData {
        name: "textRight",
        source: Some("TextAlignRightIcon"),
        marks: &[
            Mark {
                shape: Shape::Path("M3 3H21"),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
            Mark {
                shape: Shape::Path("M13 9H21"),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
            Mark {
                shape: Shape::Path("M3 15H21"),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
            Mark {
                shape: Shape::Path("M13 21H21"),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
        ],
    },
    IconData {
        name: "justifyLeft",
        source: Some("TextAlignJustifyLeftIcon"),
        marks: &[
            Mark {
                shape: Shape::Path("M3 3H11"),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
            Mark {
                shape: Shape::Path("M3 9H11"),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
            Mark {
                shape: Shape::Path("M3 15H21"),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
            Mark {
                shape: Shape::Path("M3 21H21"),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
        ],
    },
    IconData {
        name: "justifyCenter",
        source: Some("TextAlignJustifyCenterIcon"),
        marks: &[
            Mark {
                shape: Shape::Path("M3 3H21"),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
            Mark {
                shape: Shape::Path("M3 9H21"),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
            Mark {
                shape: Shape::Path("M3 15H21"),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
            Mark {
                shape: Shape::Path("M3 21H21"),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
        ],
    },
    IconData {
        name: "justifyRight",
        source: Some("TextAlignJustifyRightIcon"),
        marks: &[
            Mark {
                shape: Shape::Path("M13 3H21"),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
            Mark {
                shape: Shape::Path("M13 9H21"),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
            Mark {
                shape: Shape::Path("M3 15H21"),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
            Mark {
                shape: Shape::Path("M3 21H21"),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
        ],
    },
    IconData {
        name: "unavailable",
        source: Some("UnavailableIcon"),
        marks: &[
            Mark {
                shape: Shape::Path(
                    "M2.25 12C2.25 17.5228 6.72715 22 12.25 22C17.7728 22 22.25 17.5228 22.25 12C22.25 6.47715 17.7728 2 12.25 2C6.72715 2 2.25 6.47715 2.25 12Z",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
            Mark {
                shape: Shape::Path("M5.25 5L19.25 19"),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
        ],
    },
    IconData {
        name: "undo",
        source: Some("Undo03Icon"),
        marks: &[
            Mark {
                shape: Shape::Path(
                    "M3 8H15C18.3137 8 21 10.6863 21 14C21 17.3137 18.3137 20 15 20H11",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
            Mark {
                shape: Shape::Path(
                    "M7 4L5.8462 4.87652C3.94873 6.31801 3 7.03875 3 8C3 8.96125 3.94873 9.68199 5.8462 11.1235L7 12",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
        ],
    },
    IconData {
        name: "redo",
        source: Some("Redo03Icon"),
        marks: &[
            Mark {
                shape: Shape::Path("M21 8H9C5.68629 8 3 10.6863 3 14C3 17.3137 5.68629 20 9 20H13"),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
            Mark {
                shape: Shape::Path(
                    "M17 4L18.1538 4.87652C20.0513 6.31801 21 7.03875 21 8C21 8.96125 20.0513 9.68199 18.1538 11.1235L17 12",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
        ],
    },
    IconData {
        name: "select",
        source: Some("Cursor02Icon"),
        marks: &[Mark {
            shape: Shape::Path(
                "M9.80282 4.62973L15.8364 6.99069C19.3164 8.35243 21.0564 9.03329 20.9987 10.1133C20.941 11.1934 19.1251 11.6886 15.4933 12.6791C14.412 12.974 13.8713 13.1215 13.4964 13.4963C13.1215 13.8712 12.9741 14.4119 12.6791 15.4933C11.6887 19.125 11.1934 20.9409 10.1134 20.9986C9.03335 21.0563 8.35249 19.3163 6.99075 15.8363L4.62979 9.80276C3.20411 6.15934 2.49127 4.33764 3.41448 3.41442C4.3377 2.49121 6.15941 3.20405 9.80282 4.62973Z",
            ),
            fill: false,
            even_odd: false,
            stroke: Some(Stroke {
                width: 1.5,
                cap: Cap::Butt,
                join: Join::Round,
            }),
        }],
    },
    IconData {
        name: "split",
        source: Some("SquareSplitHorizontalIcon"),
        marks: &[
            Mark {
                shape: Shape::Path("M11.9922 21L11.9922 20L11.9922 4L11.9922 3"),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
            Mark {
                shape: Shape::Path(
                    "M7.99219 19.9995C6.69208 19.8699 5.81694 19.5772 5.16376 18.9157C3.99219 17.7291 3.99219 15.8193 3.99219 11.9997C3.99219 8.18016 3.99219 6.27038 5.16376 5.08379C5.81694 4.42224 6.69207 4.12952 7.99219 4",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Miter,
                }),
            },
            Mark {
                shape: Shape::Path(
                    "M15.9922 19.9998C17.2923 19.8703 18.1674 19.5776 18.8206 18.916C19.9922 17.7294 19.9922 15.8196 19.9922 12.0001C19.9922 8.1805 19.9922 6.27072 18.8206 5.08413C18.192 4.44741 17.3577 4.15236 16.1368 4.01562",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Miter,
                }),
            },
        ],
    },
    IconData {
        name: "delete",
        source: Some("Delete02Icon"),
        marks: &[
            Mark {
                shape: Shape::Path(
                    "M18.8516 16.3743L19.5 6H4.5L5.14839 16.3743C5.315 19.0401 5.39831 20.373 6.26426 21.1865C7.13021 22 8.46571 22 11.1367 22H12.8633C15.5343 22 16.8698 22 17.7357 21.1865C18.6017 20.373 18.685 19.0401 18.8516 16.3743Z",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
            Mark {
                shape: Shape::Path(
                    "M16 6H21M16 6L15.3778 4.34082C15.0007 3.33524 14.8122 2.83245 14.4402 2.50292C14.3213 2.39765 14.1904 2.3069 14.0501 2.2326C13.6109 2 13.074 2 12 2C10.926 2 10.3891 2 9.9499 2.2326C9.8096 2.3069 9.67865 2.39765 9.55982 2.50292C9.18783 2.83245 8.99928 3.33524 8.62219 4.34082L8 6M16 6H8M3 6H8",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
            Mark {
                shape: Shape::Path("M9.5 11V17M14.5 11V17"),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
        ],
    },
    IconData {
        name: "freeze",
        source: Some("SnowflakeIcon"),
        marks: &[
            Mark {
                shape: Shape::Path(
                    "M14.244 3.00632C13.801 3.40963 12.6212 5.00086 11.991 5C11.3608 4.99909 10.1858 3.40451 9.744 3",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
            Mark {
                shape: Shape::Path(
                    "M14.244 20.9937C13.801 20.5904 12.6212 18.9991 11.991 19C11.3608 19.0009 10.1858 20.5955 9.744 21",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
            Mark {
                shape: Shape::Path(
                    "M20.9077 9.45191C20.337 9.26993 18.3691 9.04383 18.0547 8.4976C17.7404 7.9514 18.5338 6.13651 18.6632 5.55164",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
            Mark {
                shape: Shape::Path(
                    "M5.33024 18.4456C5.45802 17.8603 6.24618 16.043 5.93032 15.4976C5.61445 14.9523 3.64599 14.732 3.07477 14.5516",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
            Mark {
                shape: Shape::Path(
                    "M3.07602 9.45203C3.64679 9.27005 5.61472 9.04395 5.9291 8.49772C6.2434 7.95152 5.44996 6.13663 5.32055 5.55176",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
            Mark {
                shape: Shape::Path(
                    "M18.6529 18.4457C18.5252 17.8604 17.737 16.0431 18.0529 15.4977C18.3687 14.9524 20.3372 14.7321 20.9084 14.5518",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
            Mark {
                shape: Shape::Path(
                    "M15.4581 9.97911L17.994 8.49982M11.9937 8.02071L11.9937 4.99982M17.9937 15.4998L15.458 14.0206M5.99424 15.4998L8.56579 13.9997M11.9937 18.9998L11.9937 15.9789M8.56579 10.0001L5.994 8.49982",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
            Mark {
                shape: Shape::Path(
                    "M15.4957 11.7313C15.4955 10.8861 15.4955 10.4635 15.2945 10.1155C15.0935 9.76744 14.7275 9.55617 13.9955 9.13361L13.5316 8.86583C12.7996 8.44326 12.4336 8.23198 12.0317 8.23199C11.6298 8.23201 11.2638 8.44332 10.5318 8.86593L10.0679 9.13379C9.33581 9.55644 8.96978 9.76777 8.76882 10.1158C8.56786 10.4639 8.56786 10.8866 8.56786 11.7319L8.56786 12.2677C8.56786 13.113 8.56786 13.5356 8.76881 13.8837C8.96976 14.2318 9.33578 14.4431 10.0678 14.8658L10.5319 15.1337C11.264 15.5564 11.63 15.7678 12.032 15.7678C12.4339 15.7678 12.7999 15.5564 13.532 15.1338L13.9958 14.866C14.728 14.4433 15.0941 14.2319 15.295 13.8838C15.496 13.5356 15.4959 13.1129 15.4958 12.2674L15.4957 11.7313Z",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Butt,
                    join: Join::Round,
                }),
            },
        ],
    },
    IconData {
        name: "reverse",
        source: Some("RewindIcon"),
        marks: &[
            Mark {
                shape: Shape::Path(
                    "M2.15512 12.9178C2.43749 13.6884 3.29078 14.3047 4.99737 15.5372C7.32449 17.218 8.48806 18.0583 9.46099 17.9969C10.204 17.9499 10.8946 17.6007 11.3699 17.0315C11.9922 16.2863 11.9922 14.8575 11.9922 12C11.9922 9.14246 11.9922 7.71369 11.3699 6.96846C10.8946 6.39933 10.204 6.0501 9.46099 6.00315C8.48806 5.94167 7.32449 6.78203 4.99737 8.46275C3.29078 9.6953 2.43749 10.3116 2.15512 11.0822C1.93788 11.675 1.93788 12.325 2.15512 12.9178Z",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Butt,
                    join: Join::Round,
                }),
            },
            Mark {
                shape: Shape::Path(
                    "M14.9974 15.5372C17.3245 17.218 18.4881 18.0583 19.461 17.9969C20.204 17.9499 20.8946 17.6007 21.3699 17.0315C21.9922 16.2863 21.9922 14.8575 21.9922 12C21.9922 9.14246 21.9922 7.71369 21.3699 6.96846C20.8946 6.39933 20.204 6.0501 19.461 6.00315C18.4881 5.94167 17.3245 6.78203 14.9974 8.46275",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
        ],
    },
    IconData {
        name: "mirror",
        source: Some("FlipHorizontalIcon"),
        marks: &[
            Mark {
                shape: Shape::Path(
                    "M6.29516 14.4204C7.65364 13.4668 8.33288 12.99 8.46522 12.3436C8.51159 12.1172 8.51159 11.8828 8.46522 11.6563C8.33288 11.01 7.65364 10.5332 6.29516 9.57959C4.65002 8.42475 3.82745 7.84734 3.15323 8.03482C2.91938 8.09984 2.70324 8.22164 2.52214 8.39045C2 8.87715 2 9.9181 2 12C2 14.0819 2 15.1229 2.52214 15.6096C2.70324 15.7784 2.91938 15.9002 3.15323 15.9652C3.82745 16.1527 4.65003 15.5752 6.29516 14.4204Z",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
            Mark {
                shape: Shape::Path(
                    "M17.7048 9.57959C16.3464 10.5332 15.6671 11.01 15.5348 11.6564C15.4884 11.8828 15.4884 12.1172 15.5348 12.3436C15.6671 12.99 16.3464 13.4668 17.7048 14.4204C19.35 15.5752 20.1725 16.1527 20.8468 15.9652C21.0806 15.9002 21.2968 15.7784 21.4779 15.6096C22 15.1229 22 14.0819 22 12C22 9.9181 22 8.87715 21.4779 8.39045C21.2968 8.22164 21.0806 8.09984 20.8468 8.03482C20.1725 7.84734 19.35 8.42476 17.7048 9.57959Z",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
            Mark {
                shape: Shape::Path("M12 14V10M12 6.5V3M12 21V17.5"),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Miter,
                }),
            },
        ],
    },
    IconData {
        name: "crop",
        source: Some("CropIcon"),
        marks: &[
            Mark {
                shape: Shape::Path(
                    "M4 2V4M22 20H20M16.5 20H10C7.17157 20 5.75736 20 4.87868 19.1213C4 18.2426 4 16.8284 4 14V7.5",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
            Mark {
                shape: Shape::Path(
                    "M20 22L20 12C20 8.22877 20 6.34315 18.8284 5.17158C17.6569 4 15.7712 4 12 4L2 4",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
        ],
    },
    IconData {
        name: "transcribe",
        source: Some("SpeechToTextIcon"),
        marks: &[
            Mark {
                shape: Shape::Path("M16 17H10"),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Miter,
                }),
            },
            Mark {
                shape: Shape::Path("M8 4V8M5 2V10M2 5L2 7M11 5V7"),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Miter,
                }),
            },
            Mark {
                shape: Shape::Path(
                    "M4.00006 13C4.00172 17.1517 4.04756 19.2749 5.31802 20.6124C6.63604 22 8.75736 22 13 22H13.45C17.2568 22 19.1601 22 20.4225 20.8649C20.6018 20.7038 20.7687 20.528 20.9218 20.3393C22 19.0103 22 17.0065 22 12.9989C22 8.99125 22 6.98744 20.9218 5.65845C20.7687 5.46974 20.6018 5.29398 20.4225 5.1328C19.3191 4.14066 17.7259 4.01573 14.8 4H14",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Miter,
                }),
            },
        ],
    },
    IconData {
        name: "splitScenes",
        source: Some("ScissorRectangleIcon"),
        marks: &[
            Mark {
                shape: Shape::Path(
                    "M2.5 12C2.5 7.52166 2.5 5.28249 3.89124 3.89124C5.28249 2.5 7.52166 2.5 12 2.5C16.4783 2.5 18.7175 2.5 20.1088 3.89124C21.5 5.28249 21.5 7.52166 21.5 12C21.5 16.4783 21.5 18.7175 20.1088 20.1088C18.7175 21.5 16.4783 21.5 12 21.5C7.52166 21.5 5.28249 21.5 3.89124 20.1088C2.5 18.7175 2.5 16.4783 2.5 12Z",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Butt,
                    join: Join::Miter,
                }),
            },
            Mark {
                shape: Shape::Path(
                    "M13.5 9.5L11 12M11 12L7 16M11 12L13.5 14.5M11 12L7 8M17 8C17 9.10457 16.1046 10 15 10C13.8954 10 13 9.10457 13 8C13 6.89543 13.8954 6 15 6C16.1046 6 17 6.89543 17 8ZM17 16C17 17.1046 16.1046 18 15 18C13.8954 18 13 17.1046 13 16C13 14.8954 13.8954 14 15 14C16.1046 14 17 14.8954 17 16Z",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
        ],
    },
    IconData {
        name: "snap",
        source: Some("AlignSelectionIcon"),
        marks: &[
            Mark {
                shape: Shape::Path("M20 6V18M18 4H6M18 20H6M4 18V6"),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
            Mark {
                shape: Shape::Path(
                    "M22 4C22 5.10457 21.1046 6 20 6C18.8954 6 18 5.10457 18 4C18 2.89543 18.8954 2 20 2C21.1046 2 22 2.89543 22 4Z",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Butt,
                    join: Join::Miter,
                }),
            },
            Mark {
                shape: Shape::Path(
                    "M6 4C6 5.10457 5.10457 6 4 6C2.89543 6 2 5.10457 2 4C2 2.89543 2.89543 2 4 2C5.10457 2 6 2.89543 6 4Z",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Butt,
                    join: Join::Miter,
                }),
            },
            Mark {
                shape: Shape::Path(
                    "M22 20C22 21.1046 21.1046 22 20 22C18.8954 22 18 21.1046 18 20C18 18.8954 18.8954 18 20 18C21.1046 18 22 18.8954 22 20Z",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Butt,
                    join: Join::Miter,
                }),
            },
            Mark {
                shape: Shape::Path(
                    "M6 20C6 21.1046 5.10457 22 4 22C2.89543 22 2 21.1046 2 20C2 18.8954 2.89543 18 4 18C5.10457 18 6 18.8954 6 20Z",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Butt,
                    join: Join::Miter,
                }),
            },
        ],
    },
    IconData {
        name: "magnet",
        source: Some("MagnetIcon"),
        marks: &[
            Mark {
                shape: Shape::Path(
                    "M3.925 2.5H9.15C9.15 2.5 8.2 7.7 8.2 10.1C8.2 13.1 10.1 15.5 12 15.5C13.9 15.5 15.8 13.1 15.8 10.1C15.8 7.7 14.85 2.5 14.85 2.5H20.075C20.075 2.5 21.5 7.13842 21.5 11.5C21.5 17 17.7 21.5 12 21.5C6.3 21.5 2.5 17 2.5 11.5C2.5 7.13842 3.925 2.5 3.925 2.5Z",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Butt,
                    join: Join::Round,
                }),
            },
            Mark {
                shape: Shape::Path("M2.97266 7.25H8.19766M15.7977 7.25L21.0227 7.25"),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Miter,
                }),
            },
        ],
    },
    IconData {
        name: "linkage",
        source: Some("Link01Icon"),
        marks: &[
            Mark {
                shape: Shape::Path(
                    "M9.14339 10.691L9.35031 10.4841C11.329 8.50532 14.5372 8.50532 16.5159 10.4841C18.4947 12.4628 18.4947 15.671 16.5159 17.6497L13.6497 20.5159C11.671 22.4947 8.46279 22.4947 6.48405 20.5159C4.50532 18.5372 4.50532 15.329 6.48405 13.3503L6.9484 12.886",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Miter,
                }),
            },
            Mark {
                shape: Shape::Path(
                    "M17.0516 11.114L17.5159 10.6497C19.4947 8.67095 19.4947 5.46279 17.5159 3.48405C15.5372 1.50532 12.329 1.50532 10.3503 3.48405L7.48405 6.35031C5.50532 8.32904 5.50532 11.5372 7.48405 13.5159C9.46279 15.4947 12.671 15.4947 14.6497 13.5159L14.8566 13.309",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Miter,
                }),
            },
        ],
    },
    IconData {
        name: "marker",
        source: Some("Flag01Icon"),
        marks: &[Mark {
            shape: Shape::Path(
                "M5.0249 21C5.04385 19.2643 5.04366 17.5541 5.0366 15.9209M5.0366 15.9209C5.01301 10.4614 4.91276 5.86186 5.19475 4.04271C5.5611 1.67939 9.39301 3.82993 13.9703 5.59842L16.0328 6.48729C17.5508 7.1415 19.7187 8.30352 18.7662 9.66084C18.3738 10.22 17.56 10.8596 16.0575 11.567L5.0366 15.9209Z",
            ),
            fill: false,
            even_odd: false,
            stroke: Some(Stroke {
                width: 1.5,
                cap: Cap::Round,
                join: Join::Round,
            }),
        }],
    },
    IconData {
        name: "beat",
        source: Some("MusicNote01Icon"),
        marks: &[
            Mark {
                shape: Shape::Path(
                    "M7 9.5C7 10.8807 5.88071 12 4.5 12C3.11929 12 2 10.8807 2 9.5C2 8.11929 3.11929 7 4.5 7C5.88071 7 7 8.11929 7 9.5ZM7 9.5V2C7.33333 2.5 7.6 4.6 10 5",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
            Mark {
                shape: Shape::Circle {
                    cx: 10.5,
                    cy: 19.5,
                    r: 2.5,
                },
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Butt,
                    join: Join::Miter,
                }),
            },
            Mark {
                shape: Shape::Circle {
                    cx: 20.0,
                    cy: 18.0,
                    r: 2.0,
                },
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Butt,
                    join: Join::Miter,
                }),
            },
            Mark {
                shape: Shape::Path(
                    "M13 19.5L13 11C13 10.09 13 9.63502 13.2466 9.35248C13.4932 9.06993 13.9938 9.00163 14.9949 8.86504C18.0085 8.45385 20.2013 7.19797 21.3696 6.42937C21.6498 6.24509 21.7898 6.15295 21.8949 6.20961C22 6.26627 22 6.43179 22 6.76283V17.9259",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
            Mark {
                shape: Shape::Path("M13 13C17.8 13 21 10.6667 22 10"),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
        ],
    },
    IconData {
        name: "normalize",
        source: Some("AudioWaveformIcon"),
        marks: &[Mark {
            shape: Shape::Path(
                "M22 14H21.7276C21.348 14 21.1582 14 20.9963 13.9637C20.495 13.8514 20.086 13.4904 19.9124 13.0069C19.8563 12.8507 19.8328 12.6624 19.7857 12.2857L19.6225 10.98C19.5525 10.4201 19.0766 10 18.5124 10C17.9377 10 17.4564 10.4355 17.3993 11.0074L16.5992 19.0081C16.5429 19.5712 16.069 20 15.5031 20C14.9346 20 14.4594 19.5673 14.4064 19.0013L13.0933 4.99563C13.0404 4.43134 12.5668 4 12 4C11.4332 4 10.9596 4.43134 10.9067 4.99564L9.59363 19.0013C9.54056 19.5673 9.06543 20 8.49689 20C7.93097 20 7.45712 19.5712 7.40081 19.0081L6.60074 11.0074C6.54355 10.4355 6.06232 10 5.48758 10C4.92339 10 4.44749 10.4201 4.37751 10.98L4.21429 12.2857C4.1672 12.6624 4.14366 12.8507 4.08759 13.0069C3.91398 13.4904 3.50501 13.8514 3.00369 13.9637C2.84176 14 2.65197 14 2.27237 14H2",
            ),
            fill: false,
            even_odd: false,
            stroke: Some(Stroke {
                width: 1.5,
                cap: Cap::Round,
                join: Join::Round,
            }),
        }],
    },
    IconData {
        name: "enhanceVoice",
        source: Some("AiVoiceIcon"),
        marks: &[
            Mark {
                shape: Shape::Path("M10.5 7.5V17.5"),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
            Mark {
                shape: Shape::Path("M14.5 12.5L14.5 19.5"),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
            Mark {
                shape: Shape::Path("M18.5 12.5L18.5 14.5"),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
            Mark {
                shape: Shape::Path("M6.5 3.5V21.5"),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
            Mark {
                shape: Shape::Path("M2.5 9.5V15.5"),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
            Mark {
                shape: Shape::Path(
                    "M14.5 6.5V4.25C14.5 3.2835 15.2835 2.5 16.25 2.5C17.2165 2.5 18 3.2835 18 4.25V6.5M14.5 6.5V8.5M14.5 6.5H18M18 8.5V6.5M21.5 2.5V8.5",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
        ],
    },
    IconData {
        name: "reduceNoise",
        source: Some("VolumeMinusIcon"),
        marks: &[
            Mark {
                shape: Shape::Path(
                    "M14 14.8135V9.18646C14 6.04126 14 4.46866 13.0747 4.0773C12.1494 3.68593 11.0603 4.79793 8.88232 7.02192C7.75439 8.17365 7.11085 8.42869 5.50604 8.42869C4.10257 8.42869 3.40084 8.42869 2.89675 8.77262C1.85035 9.48655 2.00852 10.882 2.00852 12C2.00852 13.118 1.85035 14.5134 2.89675 15.2274C3.40084 15.5713 4.10257 15.5713 5.50604 15.5713C7.11085 15.5713 7.75439 15.8264 8.88232 16.9781C11.0603 19.2021 12.1494 20.3141 13.0747 19.9227C14 19.5313 14 17.9587 14 14.8135Z",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
            Mark {
                shape: Shape::Path("M17 12L22 12"),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Miter,
                }),
            },
        ],
    },
    IconData {
        name: "zoomIn",
        source: Some("ZoomInAreaIcon"),
        marks: &[
            Mark {
                shape: Shape::Path(
                    "M18.5016 19.1217L21 21.6217M20 15.1217C20 12.0842 17.5376 9.62173 14.5 9.62173C11.4624 9.62173 9 12.0842 9 15.1217C9 18.1593 11.4624 20.6217 14.5 20.6217C17.5376 20.6217 20 18.1593 20 15.1217Z",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
            Mark {
                shape: Shape::Path("M14.5 13.1217V17.1217M16.5 15.1217H12.5"),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
            Mark {
                shape: Shape::Path(
                    "M10 3.62173H14M3 10.6217V14.6217M6.5 21.6217C4.567 21.6217 3 20.0547 3 18.1217M17.5 3.62173C19.433 3.62173 21 5.18873 21 7.12173M3 7.12173C3 5.18873 4.567 3.62173 6.5 3.62173",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
        ],
    },
    IconData {
        name: "zoomOut",
        source: Some("ZoomOutAreaIcon"),
        marks: &[
            Mark {
                shape: Shape::Path(
                    "M18.5016 18.5L21 21M20 14.5C20 11.4624 17.5376 9 14.5 9C11.4624 9 9 11.4624 9 14.5C9 17.5376 11.4624 20 14.5 20C17.5376 20 20 17.5376 20 14.5Z",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
            Mark {
                shape: Shape::Path("M16.5 14.5H12.5"),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
            Mark {
                shape: Shape::Path(
                    "M10 3H14M3 10V14M6.5 21C4.567 21 3 19.433 3 17.5M17.5 3C19.433 3 21 4.567 21 6.5M3 6.5C3 4.567 4.567 3 6.5 3",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
        ],
    },
    crate::custom::ZOOMFIT,
    IconData {
        name: "guides",
        source: Some("RulerIcon"),
        marks: &[
            Mark {
                shape: Shape::Path("M17.5 10.5L19.5 12.5M14 14L16 16M10.5 17.5L12.5 19.5"),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Miter,
                }),
            },
            Mark {
                shape: Shape::Path(
                    "M10.5355 4.67767C11.9002 3.31296 12.5826 2.6306 13.3438 2.3153C14.3587 1.8949 15.4991 1.8949 16.5141 2.3153C17.2753 2.6306 17.9576 3.31296 19.3223 4.67767C20.687 6.04238 21.3694 6.72474 21.6847 7.48594C22.1051 8.50088 22.1051 9.64126 21.6847 10.6562C21.3694 11.4174 20.687 12.0998 19.3223 13.4645L13.4645 19.3223C12.0998 20.687 11.4174 21.3694 10.6562 21.6847C9.64126 22.1051 8.50088 22.1051 7.48594 21.6847C6.72474 21.3694 6.04238 20.687 4.67767 19.3223C3.31296 17.9576 2.6306 17.2753 2.3153 16.5141C1.8949 15.4991 1.8949 14.3587 2.3153 13.3438C2.6306 12.5826 3.31296 11.9002 4.67767 10.5355L10.5355 4.67767Z",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
        ],
    },
    IconData {
        name: "lock",
        source: Some("SquareLock01Icon"),
        marks: &[
            Mark {
                shape: Shape::Path("M12 14V17.0023"),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
            Mark {
                shape: Shape::Path(
                    "M16.5 9V6.5C16.5 4.01472 14.4853 2 12 2C9.51472 2 7.5 4.01472 7.5 6.5V9",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
            Mark {
                shape: Shape::Path(
                    "M4.26781 18.8447C4.49269 20.515 5.87613 21.8235 7.55966 21.9009C8.97627 21.966 10.4153 22 12 22C13.5847 22 15.0237 21.966 16.4403 21.9009C18.1239 21.8235 19.5073 20.515 19.7322 18.8447C19.879 17.7547 20 16.6376 20 15.5C20 14.3624 19.879 13.2453 19.7322 12.1553C19.5073 10.485 18.1239 9.17649 16.4403 9.09909C15.0237 9.03397 13.5847 9 12 9C10.4153 9 8.97627 9.03397 7.55966 9.09909C5.87613 9.17649 4.49269 10.485 4.26781 12.1553C4.12104 13.2453 4 14.3624 4 15.5C4 16.6376 4.12104 17.7547 4.26781 18.8447Z",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
        ],
    },
    IconData {
        name: "done",
        source: Some("Tick02Icon"),
        marks: &[Mark {
            shape: Shape::Path("M5 14L8.5 17.5L19 6.5"),
            fill: false,
            even_odd: false,
            stroke: Some(Stroke {
                width: 1.5,
                cap: Cap::Round,
                join: Join::Round,
            }),
        }],
    },
    IconData {
        name: "home",
        source: Some("Home01Icon"),
        marks: &[Mark {
            shape: Shape::Path(
                "M3 11.9896V14.5C3 17.7998 3 19.4497 4.02513 20.4749C5.05025 21.5 6.70017 21.5 10 21.5H14C17.2998 21.5 18.9497 21.5 19.9749 20.4749C21 19.4497 21 17.7998 21 14.5V11.9896C21 10.3083 21 9.46773 20.6441 8.74005C20.2882 8.01237 19.6247 7.49628 18.2976 6.46411L16.2976 4.90855C14.2331 3.30285 13.2009 2.5 12 2.5C10.7991 2.5 9.76689 3.30285 7.70242 4.90855L5.70241 6.46411C4.37533 7.49628 3.71179 8.01237 3.3559 8.74005C3 9.46773 3 10.3083 3 11.9896Z",
            ),
            fill: false,
            even_odd: false,
            stroke: Some(Stroke {
                width: 1.5,
                cap: Cap::Round,
                join: Join::Round,
            }),
        }],
    },
    IconData {
        name: "arrowRight",
        source: Some("ArrowRight02Icon"),
        marks: &[
            Mark {
                shape: Shape::Path("M18.5 12L4.99997 12"),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
            Mark {
                shape: Shape::Path("M13 18C13 18 19 13.5811 19 12C19 10.4188 13 6 13 6"),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
        ],
    },
    IconData {
        name: "scissors",
        source: Some("ScissorIcon"),
        marks: &[
            Mark {
                shape: Shape::Path(
                    "M21 6C21 7.65685 19.6569 9 18 9C16.3431 9 15 7.65685 15 6C15 4.34315 16.3431 3 18 3C19.6569 3 21 4.34315 21 6Z",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Butt,
                    join: Join::Miter,
                }),
            },
            Mark {
                shape: Shape::Path(
                    "M21 18C21 19.6569 19.6569 21 18 21C16.3431 21 15 19.6569 15 18C15 16.3431 16.3431 15 18 15C19.6569 15 21 16.3431 21 18Z",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Butt,
                    join: Join::Miter,
                }),
            },
            Mark {
                shape: Shape::Path("M15 8L3 19"),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
            Mark {
                shape: Shape::Path("M15.0003 16L11 12"),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
            Mark {
                shape: Shape::Path("M3 5.00011L8.5 10"),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
        ],
    },
    IconData {
        name: "overlay",
        source: Some("Layers01Icon"),
        marks: &[
            Mark {
                shape: Shape::Path(
                    "M8.64298 3.14559L6.93816 3.93362C4.31272 5.14719 3 5.75397 3 6.75C3 7.74603 4.31272 8.35281 6.93817 9.56638L8.64298 10.3544C10.2952 11.1181 11.1214 11.5 12 11.5C12.8786 11.5 13.7048 11.1181 15.357 10.3544L17.0618 9.56638C19.6873 8.35281 21 7.74603 21 6.75C21 5.75397 19.6873 5.14719 17.0618 3.93362L15.357 3.14559C13.7048 2.38186 12.8786 2 12 2C11.1214 2 10.2952 2.38186 8.64298 3.14559Z",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
            Mark {
                shape: Shape::Path(
                    "M20.788 11.0972C20.9293 11.2959 21 11.5031 21 11.7309C21 12.7127 19.6873 13.3109 17.0618 14.5072L15.357 15.284C13.7048 16.0368 12.8786 16.4133 12 16.4133C11.1214 16.4133 10.2952 16.0368 8.64298 15.284L6.93817 14.5072C4.31272 13.3109 3 12.7127 3 11.7309C3 11.5031 3.07067 11.2959 3.212 11.0972",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
            Mark {
                shape: Shape::Path(
                    "M20.3767 16.2661C20.7922 16.5971 21 16.927 21 17.3176C21 18.2995 19.6873 18.8976 17.0618 20.0939L15.357 20.8707C13.7048 21.6236 12.8786 22 12 22C11.1214 22 10.2952 21.6236 8.64298 20.8707L6.93817 20.0939C4.31272 18.8976 3 18.2995 3 17.3176C3 16.927 3.20778 16.5971 3.62334 16.2661",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
        ],
    },
    IconData {
        name: "captions",
        source: Some("ClosedCaptionIcon"),
        marks: &[
            Mark {
                shape: Shape::Path(
                    "M2 12C2 8.02033 2 6.03049 3.0528 4.70201C3.22119 4.48953 3.40678 4.29302 3.60746 4.11473C4.86213 3 6.74142 3 10.5 3H13.5C17.2586 3 19.1379 3 20.3925 4.11473C20.5932 4.29302 20.7788 4.48953 20.9472 4.70201C22 6.03049 22 8.02033 22 12C22 15.9797 22 17.9695 20.9472 19.298C20.7788 19.5105 20.5932 19.707 20.3925 19.8853C19.1379 21 17.2586 21 13.5 21H10.5C6.74142 21 4.86213 21 3.60746 19.8853C3.40678 19.707 3.22119 19.5105 3.0528 19.298C2 17.9695 2 15.9797 2 12Z",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Miter,
                }),
            },
            Mark {
                shape: Shape::Path(
                    "M10.5 9H10C9.06812 9 8.60218 9 8.23463 9.15224C7.74458 9.35523 7.35523 9.74458 7.15224 10.2346C7 10.6022 7 11.0681 7 12C7 12.9319 7 13.3978 7.15224 13.7654C7.35523 14.2554 7.74458 14.6448 8.23463 14.8478C8.60218 15 9.06812 15 10 15H10.5M17 9H16.5C15.5681 9 15.1022 9 14.7346 9.15224C14.2446 9.35523 13.8552 9.74458 13.6522 10.2346C13.5 10.6022 13.5 11.0681 13.5 12C13.5 12.9319 13.5 13.3978 13.6522 13.7654C13.8552 14.2554 14.2446 14.6448 14.7346 14.8478C15.1022 15 15.5681 15 16.5 15H17",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Miter,
                }),
            },
        ],
    },
    IconData {
        name: "aspectRatio",
        source: Some("AspectRatioIcon"),
        marks: &[
            Mark {
                shape: Shape::Path(
                    "M2 12C2 7.75736 2 5.63604 3.17157 4.31802C4.34315 3 6.22876 3 10 3H14C17.7712 3 19.6569 3 20.8284 4.31802C22 5.63604 22 7.75736 22 12C22 16.2426 22 18.364 20.8284 19.682C19.6569 21 17.7712 21 14 21H10C6.22876 21 4.34315 21 3.17157 19.682C2 18.364 2 16.2426 2 12Z",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Butt,
                    join: Join::Miter,
                }),
            },
            Mark {
                shape: Shape::Path(
                    "M2 9H10C12.8284 9 14.2426 9 15.1213 9.87868C16 10.7574 16 12.1716 16 15V21",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Butt,
                    join: Join::Miter,
                }),
            },
            Mark {
                shape: Shape::Path("M10 21L10 9"),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Butt,
                    join: Join::Miter,
                }),
            },
        ],
    },
    IconData {
        name: "background",
        source: Some("PaintBoardIcon"),
        marks: &[
            Mark {
                shape: Shape::Path(
                    "M22 12C22 6.47715 17.5228 2 12 2C6.47715 2 2 6.47715 2 12C2 17.5228 6.47715 22 12 22C12.8417 22 14 22.1163 14 21C14 20.391 13.6832 19.9212 13.3686 19.4544C12.9082 18.7715 12.4523 18.0953 13 17C13.6667 15.6667 14.7778 15.6667 16.4815 15.6667C17.3334 15.6667 18.3334 15.6667 19.5 15.5C21.601 15.1999 22 13.9084 22 12Z",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Butt,
                    join: Join::Miter,
                }),
            },
            Mark {
                shape: Shape::Circle {
                    cx: 9.5,
                    cy: 8.5,
                    r: 1.5,
                },
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Butt,
                    join: Join::Miter,
                }),
            },
            Mark {
                shape: Shape::Circle {
                    cx: 16.5,
                    cy: 9.5,
                    r: 1.5,
                },
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Butt,
                    join: Join::Miter,
                }),
            },
            Mark {
                shape: Shape::Path(
                    "M7.125 15H7M7.25 15C7.25 15.1381 7.13807 15.25 7 15.25C6.86193 15.25 6.75 15.1381 6.75 15C6.75 14.8619 6.86193 14.75 7 14.75C7.13807 14.75 7.25 14.8619 7.25 15Z",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
        ],
    },
    IconData {
        name: "volume",
        source: Some("VolumeHighIcon"),
        marks: &[
            Mark {
                shape: Shape::Path(
                    "M14 14.8135V9.18646C14 6.04126 14 4.46866 13.0747 4.0773C12.1494 3.68593 11.0603 4.79793 8.88232 7.02192C7.75439 8.17365 7.11085 8.42869 5.50604 8.42869C4.10257 8.42869 3.40084 8.42869 2.89675 8.77262C1.85035 9.48655 2.00852 10.882 2.00852 12C2.00852 13.118 1.85035 14.5134 2.89675 15.2274C3.40084 15.5713 4.10257 15.5713 5.50604 15.5713C7.11085 15.5713 7.75439 15.8264 8.88232 16.9781C11.0603 19.2021 12.1494 20.3141 13.0747 19.9227C14 19.5313 14 17.9587 14 14.8135Z",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
            Mark {
                shape: Shape::Path(
                    "M17 9C17.6254 9.81968 18 10.8634 18 12C18 13.1366 17.6254 14.1803 17 15",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
            Mark {
                shape: Shape::Path(
                    "M20 7C21.2508 8.36613 22 10.1057 22 12C22 13.8943 21.2508 15.6339 20 17",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
        ],
    },
    IconData {
        name: "speed",
        source: Some("DashboardSpeed01Icon"),
        marks: &[
            Mark {
                shape: Shape::Path(
                    "M13.5 13L17 9M14 15C14 16.1046 13.1046 17 12 17C10.8954 17 10 16.1046 10 15C10 13.8954 10.8954 13 12 13C13.1046 13 14 13.8954 14 15Z",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Miter,
                }),
            },
            Mark {
                shape: Shape::Path(
                    "M6 12C6 8.68629 8.68629 6 12 6C13.0929 6 14.1175 6.29218 15 6.80269",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Miter,
                }),
            },
            Mark {
                shape: Shape::Path(
                    "M2.50006 12.0001C2.50006 7.52172 2.50006 5.28255 3.8913 3.8913C5.28255 2.50006 7.52172 2.50006 12.0001 2.50006C16.4784 2.50006 18.7176 2.50006 20.1088 3.8913C21.5001 5.28255 21.5001 7.52172 21.5001 12.0001C21.5001 16.4784 21.5001 18.7176 20.1088 20.1088C18.7176 21.5001 16.4784 21.5001 12.0001 21.5001C7.52172 21.5001 5.28255 21.5001 3.8913 20.1088C2.50006 18.7176 2.50006 16.4784 2.50006 12.0001Z",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Butt,
                    join: Join::Miter,
                }),
            },
        ],
    },
    IconData {
        name: "animations",
        source: Some("Motion02Icon"),
        marks: &[
            Mark {
                shape: Shape::Path(
                    "M11 16.5C11 14.4438 11 13.4158 11.4421 12.65C11.7318 12.1483 12.1483 11.7318 12.65 11.4421C13.4158 11 14.4438 11 16.5 11C18.5562 11 19.5842 11 20.35 11.4421C20.8517 11.7318 21.2682 12.1483 21.5579 12.65C22 13.4158 22 14.4438 22 16.5C22 18.5562 22 19.5842 21.5579 20.35C21.2682 20.8517 20.8517 21.2682 20.35 21.5579C19.5842 22 18.5562 22 16.5 22C14.4438 22 13.4158 22 12.65 21.5579C12.1483 21.2682 11.7318 20.8517 11.4421 20.35C11 19.5842 11 18.5562 11 16.5Z",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Butt,
                    join: Join::Miter,
                }),
            },
            Mark {
                shape: Shape::Path(
                    "M8.51338 15C8.02667 14.9537 7.66602 14.8628 7.35723 14.6845C6.94458 14.4463 6.60191 14.1036 6.36367 13.6909C6 13.0611 6 12.2154 6 10.5241C6 8.83277 6 7.98712 6.36367 7.35723C6.60191 6.94458 6.94458 6.60191 7.35723 6.36367C7.98712 6 8.83277 6 10.5241 6C12.2154 6 13.0611 6 13.6909 6.36367C14.1036 6.60191 14.4463 6.94458 14.6845 7.35723C14.8628 7.66602 14.9537 8.02667 15 8.51338",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Miter,
                }),
            },
            Mark {
                shape: Shape::Path(
                    "M3.41611 9C2.93863 8.85544 2.53522 8.4968 2.28817 8.07057C2 7.57339 2 6.9059 2 5.57093C2 4.23595 2 3.56846 2.28817 3.07128C2.47696 2.74557 2.7485 2.4751 3.07548 2.28705C3.57462 2 4.24473 2 5.58495 2C6.92516 2 7.59527 2 8.09441 2.28705C8.49574 2.51785 8.84255 2.88838 9 3.32758",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Miter,
                }),
            },
        ],
    },
    IconData {
        name: "duplicate",
        source: Some("Copy01Icon"),
        marks: &[
            Mark {
                shape: Shape::Path(
                    "M7.5 14.5C7.5 11.2002 7.5 9.55025 8.52513 8.52513C9.55025 7.5 11.2002 7.5 14.5 7.5C17.7998 7.5 19.4497 7.5 20.4749 8.52513C21.5 9.55025 21.5 11.2002 21.5 14.5C21.5 17.7998 21.5 19.4497 20.4749 20.4749C19.4497 21.5 17.7998 21.5 14.5 21.5C11.2002 21.5 9.55025 21.5 8.52513 20.4749C7.5 19.4497 7.5 17.7998 7.5 14.5Z",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
            Mark {
                shape: Shape::Path(
                    "M7.5 16.5C6.10355 16.5 5.40533 16.5 4.84402 16.3036C3.83866 15.9518 3.0482 15.1613 2.69641 14.156C2.5 13.5947 2.5 12.8964 2.5 11.5V9.5C2.5 6.20017 2.5 4.55025 3.52513 3.52513C4.55025 2.5 6.20017 2.5 9.5 2.5H11.5C12.8964 2.5 13.5947 2.5 14.156 2.69641C15.1613 3.0482 15.9518 3.83866 16.3036 4.84402C16.5 5.40533 16.5 6.10355 16.5 7.5",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
        ],
    },
    IconData {
        name: "replace",
        source: Some("ArrowReloadHorizontalIcon"),
        marks: &[
            Mark {
                shape: Shape::Path("M20.5 5.5H9.5C5.78672 5.5 3 8.18503 3 12"),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
            Mark {
                shape: Shape::Path("M3.5 18.5H14.5C18.2133 18.5 21 15.815 21 12"),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
            Mark {
                shape: Shape::Path("M18.5 3C18.5 3 21 4.84122 21 5.50002C21 6.15882 18.5 8 18.5 8"),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
            Mark {
                shape: Shape::Path(
                    "M5.49998 16C5.49998 16 3.00001 17.8412 3 18.5C2.99999 19.1588 5.5 21 5.5 21",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
        ],
    },
    IconData {
        name: "extractAudio",
        source: Some("MusicNoteSquare01Icon"),
        marks: &[
            Mark {
                shape: Shape::Path(
                    "M2.5 12C2.5 7.52166 2.5 5.28249 3.89124 3.89124C5.28249 2.5 7.52166 2.5 12 2.5C16.4783 2.5 18.7175 2.5 20.1088 3.89124C21.5 5.28249 21.5 7.52166 21.5 12C21.5 16.4783 21.5 18.7175 20.1088 20.1088C18.7175 21.5 16.4783 21.5 12 21.5C7.52166 21.5 5.28249 21.5 3.89124 20.1088C2.5 18.7175 2.5 16.4783 2.5 12Z",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Butt,
                    join: Join::Miter,
                }),
            },
            Mark {
                shape: Shape::Path(
                    "M13 14.5C13 15.8807 11.8807 17 10.5 17C9.11929 17 8 15.8807 8 14.5C8 13.1193 9.11929 12 10.5 12C11.8807 12 13 13.1193 13 14.5ZM13 14.5V7C13.3333 7.5 13.6 9.6 16 10",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
        ],
    },
    IconData {
        name: "fade",
        source: Some("VolumeLowIcon"),
        marks: &[
            Mark {
                shape: Shape::Path(
                    "M19 9C19.6254 9.81968 20 10.8634 20 12C20 13.1366 19.6254 14.1803 19 15",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
            Mark {
                shape: Shape::Path(
                    "M16 14.8135V9.18646C16 6.04126 16 4.46866 15.0747 4.0773C14.1494 3.68593 13.0604 4.79793 10.8823 7.02192C9.7544 8.17365 9.11086 8.42869 7.50605 8.42869C6.10259 8.42869 5.40086 8.42869 4.89677 8.77262C3.85036 9.48655 4.00854 10.882 4.00854 12C4.00854 13.118 3.85036 14.5134 4.89677 15.2274C5.40086 15.5713 6.10259 15.5713 7.50605 15.5713C9.11086 15.5713 9.7544 15.8264 10.8823 16.9781C13.0604 19.2021 14.1494 20.3141 15.0747 19.9227C16 19.5313 16 17.9587 16 14.8135Z",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
        ],
    },
    IconData {
        name: "unlink",
        source: Some("Unlink01Icon"),
        marks: &[
            Mark {
                shape: Shape::Path(
                    "M12.1434 10.691L12.3503 10.4841C14.329 8.50532 17.5372 8.50532 19.5159 10.4841C21.4947 12.4628 21.4947 15.671 19.5159 17.6497L16.6497 20.5159C14.671 22.4947 11.4628 22.4947 9.48405 20.5159C7.50532 18.5372 7.50532 15.329 9.48405 13.3503L9.9484 12.886",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Miter,
                }),
            },
            Mark {
                shape: Shape::Path(
                    "M20.0516 11.114L20.5159 10.6497C22.4947 8.67095 22.4947 5.46279 20.5159 3.48405C18.5372 1.50532 15.329 1.50532 13.3503 3.48405L10.4841 6.35031C8.50532 8.32904 8.50532 11.5372 10.4841 13.5159C12.4628 15.4947 15.671 15.4947 17.6497 13.5159L17.8566 13.309",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Miter,
                }),
            },
            Mark {
                shape: Shape::Path("M4.5 4L6 6M2 8L5 9M3 13.5L5 12"),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
        ],
    },
    IconData {
        name: "splice",
        source: Some("ArrowDownToLineIcon"),
        marks: &[
            Mark {
                shape: Shape::Path("M12 16V3"),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
            Mark {
                shape: Shape::Path("M6 21H18"),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Miter,
                }),
            },
            Mark {
                shape: Shape::Path("M7 12C7 12 10.6824 17 12 17C13.3176 17 17 12 17 12"),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
        ],
    },
    IconData {
        name: "addOverlay",
        source: Some("ImageAdd01Icon"),
        marks: &[
            Mark {
                shape: Shape::Path(
                    "M11.5085 2.9903C7.02567 2.9903 4.78428 2.9903 3.39164 4.38238C1.99902 5.77447 1.99902 8.015 1.99902 12.4961C1.99902 16.9771 1.99902 19.2176 3.39164 20.6098C4.78428 22.0018 7.02567 22.0018 11.5085 22.0018C15.9912 22.0018 18.2326 22.0018 19.6253 20.6098C21.0179 19.2176 21.0179 16.9771 21.0179 12.4961V11.9958",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Miter,
                }),
            },
            Mark {
                shape: Shape::Path("M4.99902 20.9898C9.209 16.2385 13.9402 9.93727 20.999 14.6632"),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Butt,
                    join: Join::Miter,
                }),
            },
            Mark {
                shape: Shape::Path("M17.9958 1.99829V10.0064M22.0014 5.97728L13.9902 5.99217"),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
        ],
    },
    IconData {
        name: "importSound",
        source: Some("FileMusicIcon"),
        marks: &[
            Mark {
                shape: Shape::Path(
                    "M19.9995 11.0039C19.9995 11.0039 19.9995 9.43383 19.8473 9.06613C19.695 8.69843 19.406 8.40927 18.8279 7.83096L14.0914 3.09236C13.5925 2.59325 13.3431 2.3437 13.034 2.19583C12.9698 2.16508 12.9039 2.13778 12.8367 2.11406C12.5137 2 12.1609 2 11.4554 2C8.21033 2 6.58782 2 5.48884 2.88646C5.26683 3.06554 5.06459 3.26787 4.88559 3.48998C3.99951 4.58943 3.99951 6.21265 3.99951 9.45908V14.0052C3.99951 17.7781 3.99951 19.6645 5.17108 20.8366C6.11417 21.7801 7.51994 21.9641 9.99951 22M12.9995 2.50022V3.00043C12.9995 5.83009 12.9995 7.24492 13.8782 8.12398C14.7569 9.00304 16.1711 9.00304 18.9995 9.00304H19.4995",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
            Mark {
                shape: Shape::Path(
                    "M17.3966 19.7446C17.3966 20.9885 16.4123 21.9969 15.198 21.9969C13.9838 21.9969 12.9995 20.9885 12.9995 19.7446C12.9995 18.5007 13.9838 17.4923 15.198 17.4923C16.4123 17.4923 17.3966 18.5007 17.3966 19.7446ZM17.3966 19.7446V14C17.6167 14.4963 18.7036 16.6999 20.0005 16.6999",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
        ],
    },
    IconData {
        name: "record",
        source: Some("Mic01Icon"),
        marks: &[
            Mark {
                shape: Shape::Path(
                    "M7 6.5C7 4.01472 9.01472 2 11.5 2C13.9853 2 16 4.01472 16 6.5V11.5C16 13.9853 13.9853 16 11.5 16C9.01472 16 7 13.9853 7 11.5V6.5Z",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
            Mark {
                shape: Shape::Path(
                    "M11.5 19H11.0828C7.57267 19 4.57706 16.4623 4 13M11.5 19H11.9172C15.4273 19 18.4229 16.4623 19 13M11.5 19V22",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
        ],
    },
    IconData {
        name: "addText",
        source: Some("TextCreationIcon"),
        marks: &[
            Mark {
                shape: Shape::Path("M20 18V6M6 20H18M18 4H6M4 6V18"),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
            Mark {
                shape: Shape::Path(
                    "M15.9138 10C16.1887 8.30534 14.9534 8.06735 11.9564 8C8.73403 8.05948 7.70512 8.43128 7.99901 10M11.9564 8V16M10.4724 16H13.4405",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
            Mark {
                shape: Shape::Path(
                    "M21 2H19C18.4477 2 18 2.44772 18 3V5C18 5.55228 18.4477 6 19 6H21C21.5523 6 22 5.55228 22 5V3C22 2.44772 21.5523 2 21 2Z",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Butt,
                    join: Join::Miter,
                }),
            },
            Mark {
                shape: Shape::Path(
                    "M5 2H3C2.44772 2 2 2.44772 2 3V5C2 5.55228 2.44772 6 3 6H5C5.55228 6 6 5.55228 6 5V3C6 2.44772 5.55228 2 5 2Z",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Butt,
                    join: Join::Miter,
                }),
            },
            Mark {
                shape: Shape::Path(
                    "M21 18H19C18.4477 18 18 18.4477 18 19V21C18 21.5523 18.4477 22 19 22H21C21.5523 22 22 21.5523 22 21V19C22 18.4477 21.5523 18 21 18Z",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Butt,
                    join: Join::Miter,
                }),
            },
            Mark {
                shape: Shape::Path(
                    "M5 18H3C2.44772 18 2 18.4477 2 19V21C2 21.5523 2.44772 22 3 22H5C5.55228 22 6 21.5523 6 21V19C6 18.4477 5.55228 18 5 18Z",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Butt,
                    join: Join::Miter,
                }),
            },
        ],
    },
    IconData {
        name: "editText",
        source: Some("PencilEdit01Icon"),
        marks: &[
            Mark {
                shape: Shape::Path(
                    "M15.2141 5.98239L16.6158 4.58063C17.39 3.80646 18.6452 3.80646 19.4194 4.58063C20.1935 5.3548 20.1935 6.60998 19.4194 7.38415L18.0176 8.78591M15.2141 5.98239L6.98023 14.2163C5.93493 15.2616 5.41226 15.7842 5.05637 16.4211C4.70047 17.058 4.3424 18.5619 4 20C5.43809 19.6576 6.94199 19.2995 7.57889 18.9436C8.21579 18.5877 8.73844 18.0651 9.78375 17.0198L18.0176 8.78591M15.2141 5.98239L18.0176 8.78591",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
            Mark {
                shape: Shape::Path("M11 20H17"),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Miter,
                }),
            },
        ],
    },
    IconData {
        name: "draw",
        source: Some("PaintBrush01Icon"),
        marks: &[
            Mark {
                shape: Shape::Path(
                    "M3.89089 20.8727L3 21L3.12727 20.1091C3.32086 18.754 3.41765 18.0764 3.71832 17.4751C4.01899 16.8738 4.50296 16.3898 5.47091 15.4218L16.9827 3.91009C17.4062 3.48654 17.618 3.27476 17.8464 3.16155C18.2811 2.94615 18.7914 2.94615 19.2261 3.16155C19.4546 3.27476 19.6663 3.48654 20.0899 3.91009C20.5135 4.33365 20.7252 4.54543 20.8385 4.77389C21.0539 5.20856 21.0539 5.71889 20.8385 6.15356C20.7252 6.38201 20.5135 6.59379 20.0899 7.01735L8.57816 18.5291C7.61022 19.497 7.12625 19.981 6.52491 20.2817C5.92357 20.5823 5.246 20.6791 3.89089 20.8727Z",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
            Mark {
                shape: Shape::Path("M6 15L9 18M8.5 12.5L11.5 15.5"),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
        ],
    },
    IconData {
        name: "export",
        source: Some("Upload04Icon"),
        marks: &[
            Mark {
                shape: Shape::Path(
                    "M5.25 21H18.75C18.9822 21 19.0983 21 19.196 20.9904C20.1456 20.8968 20.8968 20.1456 20.9904 19.196C21 19.0983 21 18.9822 21 18.75C21 18.5178 21 18.4017 20.9904 18.304C20.8968 17.3544 20.1456 16.6032 19.196 16.5096C19.0983 16.5 18.9822 16.5 18.75 16.5H17.0607C16.8324 16.5 16.7182 16.5 16.6087 16.5121C16.2317 16.5538 15.8742 16.7018 15.5781 16.939C15.4921 17.0079 15.4114 17.0886 15.25 17.25C15.0886 17.4114 15.0079 17.4921 14.9219 17.561C14.6258 17.7982 14.2683 17.9462 13.8913 17.9879C13.7818 18 13.6676 18 13.4393 18H10.5607C10.3324 18 10.2182 18 10.1087 17.9879C9.73165 17.9462 9.37423 17.7982 9.07814 17.561C8.99213 17.4921 8.91142 17.4114 8.75 17.25C8.58858 17.0886 8.50787 17.0079 8.42186 16.939C8.12577 16.7018 7.76835 16.5538 7.3913 16.5121C7.28177 16.5 7.16763 16.5 6.93934 16.5H5.25C5.01783 16.5 4.90175 16.5 4.80397 16.5096C3.85441 16.6032 3.10315 17.3544 3.00963 18.304C3 18.4017 3 18.5178 3 18.75C3 18.9822 3 19.0983 3.00963 19.196C3.10315 20.1456 3.85441 20.8968 4.80397 20.9904C4.90175 21 5.01783 21 5.25 21Z",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
            Mark {
                shape: Shape::Path(
                    "M16.5 7.49996C16.5 7.49996 13.1858 3.00001 12 3C10.8141 2.99999 7.5 7.5 7.5 7.5M12 4V15",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
        ],
    },
    IconData {
        name: "close",
        source: Some("Cancel01Icon"),
        marks: &[
            Mark {
                shape: Shape::Path("M18.9991 19L5 5.00098"),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
            Mark {
                shape: Shape::Path("M19 5L5.00092 18.999"),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
        ],
    },
    IconData {
        name: "saved",
        source: Some("CheckmarkCircle02Icon"),
        marks: &[
            Mark {
                shape: Shape::Path(
                    "M22 12C22 6.47715 17.5228 2 12 2C6.47715 2 2 6.47715 2 12C2 17.5228 6.47715 22 12 22C17.5228 22 22 17.5228 22 12Z",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Butt,
                    join: Join::Miter,
                }),
            },
            Mark {
                shape: Shape::Path("M8 12.5L10.5 15L16 9"),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
        ],
    },
    IconData {
        name: "share",
        source: Some("Share08Icon"),
        marks: &[
            Mark {
                shape: Shape::Path(
                    "M21 6.5C21 8.15685 19.6569 9.5 18 9.5C16.3431 9.5 15 8.15685 15 6.5C15 4.84315 16.3431 3.5 18 3.5C19.6569 3.5 21 4.84315 21 6.5Z",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Butt,
                    join: Join::Miter,
                }),
            },
            Mark {
                shape: Shape::Path(
                    "M9 12C9 13.6569 7.65685 15 6 15C4.34315 15 3 13.6569 3 12C3 10.3431 4.34315 9 6 9C7.65685 9 9 10.3431 9 12Z",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Butt,
                    join: Join::Miter,
                }),
            },
            Mark {
                shape: Shape::Path(
                    "M21 17.5C21 19.1569 19.6569 20.5 18 20.5C16.3431 20.5 15 19.1569 15 17.5C15 15.8431 16.3431 14.5 18 14.5C19.6569 14.5 21 15.8431 21 17.5Z",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Butt,
                    join: Join::Miter,
                }),
            },
            Mark {
                shape: Shape::Path("M8.72852 10.7495L15.2285 7.75M8.72852 13.25L15.2285 16.2495"),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Butt,
                    join: Join::Miter,
                }),
            },
        ],
    },
    IconData {
        name: "unlock",
        source: Some("SquareUnlock01Icon"),
        marks: &[
            Mark {
                shape: Shape::Path("M12 14V17.0023"),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
            Mark {
                shape: Shape::Path(
                    "M7.5 9V6.5C7.5 4.01472 9.51472 2 12 2C13.9593 2 15.8664 3.25221 16.5 5",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
            Mark {
                shape: Shape::Path(
                    "M4.26781 18.8447C4.49269 20.515 5.87613 21.8235 7.55966 21.9009C8.97627 21.966 10.4153 22 12 22C13.5847 22 15.0237 21.966 16.4403 21.9009C18.1239 21.8235 19.5073 20.515 19.7322 18.8447C19.879 17.7547 20 16.6376 20 15.5C20 14.3624 19.879 13.2453 19.7322 12.1553C19.5073 10.485 18.1239 9.17649 16.4403 9.09909C15.0237 9.03397 13.5847 9 12 9C10.4153 9 8.97627 9.03397 7.55965 9.09909C5.87613 9.17649 4.49269 10.485 4.26781 12.1553C4.12104 13.2453 4 14.3624 4 15.5C4 16.6376 4.12104 17.7547 4.26781 18.8447Z",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
        ],
    },
    IconData {
        name: "view",
        source: Some("ViewIcon"),
        marks: &[
            Mark {
                shape: Shape::Path(
                    "M21.544 11.045C21.848 11.4713 22 11.6845 22 12C22 12.3155 21.848 12.5287 21.544 12.955C20.1779 14.8706 16.6892 19 12 19C7.31078 19 3.8221 14.8706 2.45604 12.955C2.15201 12.5287 2 12.3155 2 12C2 11.6845 2.15201 11.4713 2.45604 11.045C3.8221 9.12944 7.31078 5 12 5C16.6892 5 20.1779 9.12944 21.544 11.045Z",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Butt,
                    join: Join::Miter,
                }),
            },
            Mark {
                shape: Shape::Path(
                    "M15 12C15 10.3431 13.6569 9 12 9C10.3431 9 9 10.3431 9 12C9 13.6569 10.3431 15 12 15C13.6569 15 15 13.6569 15 12Z",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Butt,
                    join: Join::Miter,
                }),
            },
        ],
    },
    IconData {
        name: "viewOff",
        source: Some("ViewOffSlashIcon"),
        marks: &[
            Mark {
                shape: Shape::Path(
                    "M19.439 15.439C20.3636 14.5212 21.0775 13.6091 21.544 12.955C21.848 12.5287 22 12.3155 22 12C22 11.6845 21.848 11.4713 21.544 11.045C20.1779 9.12944 16.6892 5 12 5C11.0922 5 10.2294 5.15476 9.41827 5.41827M6.74742 6.74742C4.73118 8.1072 3.24215 9.94266 2.45604 11.045C2.15201 11.4713 2 11.6845 2 12C2 12.3155 2.15201 12.5287 2.45604 12.955C3.8221 14.8706 7.31078 19 12 19C13.9908 19 15.7651 18.2557 17.2526 17.2526",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
            Mark {
                shape: Shape::Path(
                    "M9.85786 10C9.32783 10.53 9 11.2623 9 12.0711C9 13.6887 10.3113 15 11.9289 15C12.7377 15 13.47 14.6722 14 14.1421",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Miter,
                }),
            },
            Mark {
                shape: Shape::Path("M3 3L21 21"),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
        ],
    },
    IconData {
        name: "volumeOff",
        source: Some("VolumeOffIcon"),
        marks: &[
            Mark {
                shape: Shape::Path("M22 22L2 2"),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
            Mark {
                shape: Shape::Path(
                    "M17 10C17.6296 10.7667 18 11.7054 18 12.7195C18 13.1635 17.929 13.593 17.7963 14",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
            Mark {
                shape: Shape::Path(
                    "M20 8C21.2508 9.22951 22 10.7952 22 12.5C22 13.9164 21.4829 15.2367 20.5906 16.348",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
            Mark {
                shape: Shape::Path(
                    "M14 14C14 17.1452 14 19.5313 13.074 19.9227C12.1481 20.3141 11.0583 19.2021 8.8787 16.9781C7.7499 15.8264 7.106 15.5713 5.5 15.5713C4.3879 15.5713 3.02749 15.7187 2.33706 14.6643C2 14.1496 2 13.4331 2 12C2 10.5669 2 9.85038 2.33706 9.33566C3.02749 8.28131 4.3879 8.42869 5.5 8.42869C6.60725 8.42869 7.3569 8.43869 7.96 7.96M14 9.5C14 6.3548 14.026 4.46866 13.1 4.0773C12.3292 3.75147 11.5323 4.46765 10 6",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
        ],
    },
    IconData {
        name: "more",
        source: Some("MoreHorizontalIcon"),
        marks: &[Mark {
            shape: Shape::Path(
                "M6.00449 12.5V12M18.0045 12.5V12M12.0045 12.5V12M7.00449 12.5C7.00449 11.9477 6.55677 11.5 6.00449 11.5C5.4522 11.5 5.00449 11.9477 5.00449 12.5C5.00449 13.0523 5.4522 13.5 6.00449 13.5C6.55677 13.5 7.00449 13.0523 7.00449 12.5ZM19.0045 12.5C19.0045 11.9477 18.5568 11.5 18.0045 11.5C17.4522 11.5 17.0045 11.9477 17.0045 12.5C17.0045 13.0523 17.4522 13.5 18.0045 13.5C18.5568 13.5 19.0045 13.0523 19.0045 12.5ZM13.0045 12.5C13.0045 11.9477 12.5568 11.5 12.0045 11.5C11.4522 11.5 11.0045 11.9477 11.0045 12.5C11.0045 13.0523 11.4522 13.5 12.0045 13.5C12.5568 13.5 13.0045 13.0523 13.0045 12.5Z",
            ),
            fill: false,
            even_odd: false,
            stroke: Some(Stroke {
                width: 1.5,
                cap: Cap::Round,
                join: Join::Round,
            }),
        }],
    },
    IconData {
        name: "solo",
        source: Some("HeadphonesIcon"),
        marks: &[
            Mark {
                shape: Shape::Path(
                    "M20.0849 17C20.5849 15.5 21 13.4368 21 12C21 7.02944 16.9706 3 12 3C7.02944 3 3 7.02944 3 12C3 13.4368 3.41512 15.5 3.91512 17",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
            Mark {
                shape: Shape::Path(
                    "M8.97651 19.6043L7.23857 14.6127C7.05341 14.1466 6.4617 13.9131 5.97493 14.0297C4.46441 14.5333 3.6462 16.1718 4.14742 17.6895L4.58543 19.0158C5.08664 20.5334 6.71747 21.3555 8.22799 20.8519C8.68896 20.6556 9.10449 20.0897 8.97651 19.6043Z",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
            Mark {
                shape: Shape::Path(
                    "M15.0235 19.6043L16.7614 14.6127C16.9466 14.1466 17.5383 13.9131 18.0251 14.0297C19.5356 14.5333 20.3538 16.1718 19.8526 17.6895L19.4146 19.0158C18.9134 20.5334 17.2825 21.3555 15.772 20.8519C15.311 20.6556 14.8955 20.0897 15.0235 19.6043Z",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
        ],
    },
    IconData {
        name: "keyCommand",
        source: Some("CommandIcon"),
        marks: &[
            Mark {
                shape: Shape::Path("M15 9V15H9V9H15Z"),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Butt,
                    join: Join::Round,
                }),
            },
            Mark {
                shape: Shape::Path(
                    "M15 15H18C19.6569 15 21 16.3431 21 18C21 19.6569 19.6569 21 18 21C16.3431 21 15 19.6569 15 18V15Z",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Butt,
                    join: Join::Round,
                }),
            },
            Mark {
                shape: Shape::Path(
                    "M9 15.002H6C4.34315 15.002 3 16.3451 3 18.002C3 19.6588 4.34315 21.002 6 21.002C7.65685 21.002 9 19.6588 9 18.002V15.002Z",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Butt,
                    join: Join::Round,
                }),
            },
            Mark {
                shape: Shape::Path(
                    "M15 9L15 6C15 4.34315 16.3431 3 18 3C19.6569 3 21 4.34315 21 6C21 7.65685 19.6569 9 18 9H15Z",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Butt,
                    join: Join::Round,
                }),
            },
            Mark {
                shape: Shape::Path(
                    "M9 9V6C9 4.34315 7.65685 3 6 3C4.34315 3 3 4.34315 3 6C3 7.65685 4.34315 9 6 9H9Z",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Butt,
                    join: Join::Round,
                }),
            },
        ],
    },
    IconData {
        name: "keyShift",
        source: Some("ArrowUpBigIcon"),
        marks: &[Mark {
            shape: Shape::Path(
                "M15.544 11.5H15V17.5C15 18.4319 15 18.8978 14.8478 19.2654C14.6448 19.7554 14.2554 20.1448 13.7654 20.3478C13.3978 20.5 12.9319 20.5 12 20.5C11.0681 20.5 10.6022 20.5 10.2346 20.3478C9.74458 20.1448 9.35523 19.7554 9.15224 19.2654C9 18.8978 9 18.4319 9 17.5V11.5H8.45596C6.37322 11.5 5.33185 11.5 5.05779 10.8997C4.78372 10.2994 5.49744 9.58174 6.92487 8.14642L10.4689 4.58281C11.1868 3.86094 11.5458 3.5 12 3.5C12.4542 3.5 12.8132 3.86094 13.5311 4.58281L17.0751 8.14643C18.5026 9.58175 19.2163 10.2994 18.9422 10.8997C18.6681 11.5 17.6268 11.5 15.544 11.5Z",
            ),
            fill: false,
            even_odd: false,
            stroke: Some(Stroke {
                width: 1.5,
                cap: Cap::Round,
                join: Join::Round,
            }),
        }],
    },
    IconData {
        name: "keyOption",
        source: Some("OptionIcon"),
        marks: &[
            Mark {
                shape: Shape::Path(
                    "M4 5H5.04362C6.96246 5 7.92187 5 8.67328 5.49548C9.42469 5.99095 9.80263 6.8728 10.5585 8.63648L13.4415 15.3635C14.1974 17.1272 14.5753 18.009 15.3267 18.5045C16.0781 19 17.0375 19 18.9564 19H20",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Miter,
                }),
            },
            Mark {
                shape: Shape::Path("M14 5H20"),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Miter,
                }),
            },
        ],
    },
    IconData {
        name: "keyWindows",
        source: Some("WindowsOldIcon"),
        marks: &[
            Mark {
                shape: Shape::Path(
                    "M14.0136 3.99538L8.01361 4.99036C5.61912 5.38744 4.42188 5.58597 3.71094 6.421C3 7.25602 3 8.46368 3 10.879L3 13.121C3 15.5363 3 16.744 3.71094 17.579C4.42188 18.414 5.61913 18.6126 8.01361 19.0096L14.0136 20.0046C17.2567 20.5424 18.8782 20.8113 19.9391 19.9171C21 19.023 21 17.3873 21 14.116V9.88402C21 6.6127 21 4.97704 19.9391 4.08286C18.8782 3.18868 17.2567 3.45758 14.0136 3.99538Z",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
            Mark {
                shape: Shape::Path("M11 4.5V19.5M3 12H21"),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
        ],
    },
    crate::custom::KEYBACKSPACE,
    crate::custom::SPLITAT,
    crate::custom::TRIMLEFT,
    crate::custom::TRIMRIGHT,
    IconData {
        name: "opacity",
        source: Some("TransparencyIcon"),
        marks: &[
            Mark {
                shape: Shape::Path(
                    "M16 9C16 12.866 12.866 16 9 16C5.13401 16 2 12.866 2 9C2 5.13401 5.13401 2 9 2C12.866 2 16 5.13401 16 9Z",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Butt,
                    join: Join::Miter,
                }),
            },
            Mark {
                shape: Shape::Path(
                    "M8.02593 15.6067C8.00876 15.4067 8 15.2044 8 15C8 11.134 11.134 8 15 8C15.2867 8 15.5693 8.01723 15.8469 8.05071M18.9969 9.25251C20.8121 10.5172 22 12.6199 22 15C22 18.866 18.866 22 15 22C12.6199 22 10.5172 20.8121 9.25251 18.9969",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
            Mark {
                shape: Shape::Path("M6.5 2.5L15.5 11.5M2.5 6.5L11.5 15.5"),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Butt,
                    join: Join::Miter,
                }),
            },
        ],
    },
    IconData {
        name: "fullScreen",
        source: Some("FullScreenIcon"),
        marks: &[Mark {
            shape: Shape::Path(
                "M15.5 21C16.8956 21 17.5933 21 18.1611 20.8278C19.4395 20.44 20.44 19.4395 20.8278 18.1611C21 17.5933 21 16.8956 21 15.5M21 8.5C21 7.10444 21 6.40666 20.8278 5.83886C20.44 4.56046 19.4395 3.56004 18.1611 3.17224C17.5933 3 16.8956 3 15.5 3M8.5 21C7.10444 21 6.40666 21 5.83886 20.8278C4.56046 20.44 3.56004 19.4395 3.17224 18.1611C3 17.5933 3 16.8956 3 15.5M3 8.5C3 7.10444 3 6.40666 3.17224 5.83886C3.56004 4.56046 4.56046 3.56004 5.83886 3.17224C6.40666 3 7.10444 3 8.5 3",
            ),
            fill: false,
            even_odd: false,
            stroke: Some(Stroke {
                width: 1.5,
                cap: Cap::Round,
                join: Join::Round,
            }),
        }],
    },
    IconData {
        name: "shapes",
        source: Some("ShapesIcon"),
        marks: &[Mark {
            shape: Shape::Path(
                "M4 8C6.20914 8 8 6.20914 8 4C8 3.2722 8 2.90829 8.02385 2.80107C8.13266 2.31172 8.28121 2.16103 8.76894 2.0452C8.87582 2.01982 9.08666 2.01679 9.50835 2.01074C10.2571 2 11.084 2 12 2C13.3714 2 14.5433 2 15.5518 2.03606C15.9598 2.05065 16.1639 2.05794 16.2867 2.09469C16.7096 2.22123 16.8563 2.37333 16.9677 2.80046C17 2.92452 17 3.28301 17 4C17 6.20914 18.7909 8 21 8C21.4931 8 21.963 8.34313 21.9759 8.83609C22 9.75428 22 10.8009 22 12C22 12.916 22 13.7429 21.9893 14.4916C21.9832 14.9133 21.9802 15.1242 21.9548 15.2311C21.839 15.7188 21.6883 15.8673 21.1989 15.9762C21.0917 16 20.7278 16 20 16C17.7909 16 16 17.7909 16 20C16 20.7278 16 21.0917 15.9762 21.1989C15.8673 21.6883 15.7188 21.839 15.2311 21.9548C15.1242 21.9802 14.9133 21.9832 14.4917 21.9893C13.7429 22 12.916 22 12 22C11.084 22 10.2571 22 9.50835 21.9893C9.08666 21.9832 8.87582 21.9802 8.76894 21.9548C8.28121 21.839 8.13266 21.6883 8.02385 21.1989C8 21.0917 8 20.7278 8 20C8 17.7909 6.20914 16 4 16C3.27219 16 2.90829 16 2.80106 15.9762C2.31172 15.8673 2.16103 15.7188 2.0452 15.2311C2.01982 15.1242 2.01679 14.9133 2.01074 14.4916C2 13.7429 2 12.916 2 12C2 11.084 2 10.2571 2.01074 9.50835C2.01679 9.08666 2.01982 8.87582 2.0452 8.76894C2.16103 8.28121 2.31172 8.13266 2.80107 8.02385C2.90829 8 3.2722 8 4 8Z",
            ),
            fill: false,
            even_odd: false,
            stroke: Some(Stroke {
                width: 1.5,
                cap: Cap::Butt,
                join: Join::Round,
            }),
        }],
    },
    IconData {
        name: "pen",
        source: Some("Pen01Icon"),
        marks: &[
            Mark {
                shape: Shape::Path(
                    "M3.49977 18.9853V20.5H5.01449C6.24074 20.5 6.85387 20.5 7.40518 20.2716C7.9565 20.0433 8.39004 19.6097 9.25713 18.7426L19.1211 8.87868C20.0037 7.99612 20.4449 7.55483 20.4937 7.01325C20.5018 6.92372 20.5018 6.83364 20.4937 6.74411C20.4449 6.20253 20.0037 5.76124 19.1211 4.87868C18.2385 3.99612 17.7972 3.55483 17.2557 3.50605C17.1661 3.49798 17.0761 3.49798 16.9865 3.50605C16.4449 3.55483 16.0037 3.99612 15.1211 4.87868L5.25713 14.7426C4.39004 15.6097 3.9565 16.0433 3.72813 16.5946C3.49977 17.1459 3.49977 17.759 3.49977 18.9853Z",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
            Mark {
                shape: Shape::Path("M13.5 6.5L17.5 10.5"),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
        ],
    },
    IconData {
        name: "highlighter",
        source: Some("HighlighterIcon"),
        marks: &[Mark {
            shape: Shape::Path(
                "M6.6777 16.2071L8.79289 18.3223M6.6777 16.2071L2.5 20.5H6.5L8.79289 18.3223M6.6777 16.2071C6.28717 15.8166 6.29534 15.1872 6.63537 14.752C7.42742 13.7383 7.71531 12.8216 7.79924 12.1382C7.89158 11.3863 8.07366 10.5734 8.60933 10.0377L9.50122 9.14828M8.79289 18.3223C9.18342 18.7128 9.81278 18.7047 10.248 18.3646C11.2617 17.5726 12.1784 17.2847 12.8618 17.2008C13.6137 17.1084 14.4266 16.9263 14.9623 16.3907L15.8517 15.4988M15.8517 15.4988L9.50122 9.14828M15.8517 15.4988C16.2422 15.8893 16.8754 15.8893 17.2659 15.4988L21.5 11.2647M9.50122 9.14828C9.1107 8.75776 9.1107 8.12459 9.50122 7.73407L13.7353 3.5",
            ),
            fill: false,
            even_odd: false,
            stroke: Some(Stroke {
                width: 1.5,
                cap: Cap::Round,
                join: Join::Round,
            }),
        }],
    },
    IconData {
        name: "eraser",
        source: Some("EraserIcon"),
        marks: &[
            Mark {
                shape: Shape::Path(
                    "M8.73792 7.78021L4.84693 11.7574C3.67722 12.953 3.09236 13.5508 3.01263 14.2802C2.99579 14.4343 2.99579 14.5899 3.01263 14.744C3.09236 15.4733 3.67722 16.0711 4.84693 17.2668L4.99601 17.4191C5.62049 18.0575 5.93274 18.3766 6.30638 18.5911C6.5236 18.7157 6.75482 18.8134 6.99505 18.882C7.40827 19 7.85149 19 8.73792 19C9.62436 19 10.0676 19 10.4808 18.882C10.721 18.8134 10.9522 18.7157 11.1695 18.5911C11.5431 18.3766 11.8554 18.0575 12.4798 17.4191L15.3239 14.5121M8.73792 7.78021L12.3199 4.12313C13.7065 2.70754 14.3997 1.99974 15.2627 2C16.1256 2.00026 16.8185 2.70846 18.2042 4.12487L18.9473 4.8845C20.3159 6.28342 21.0002 6.98288 21 7.85008C20.9997 8.71728 20.315 9.41633 18.9456 10.8144L15.3239 14.5121M8.73792 7.78021L15.3239 14.5121",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
            Mark {
                shape: Shape::Path("M10 22L21 22"),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
        ],
    },
    IconData {
        name: "letterSpacing",
        source: Some("LetterSpacingIcon"),
        marks: &[
            Mark {
                shape: Shape::Path("M2 22V2"),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Miter,
                }),
            },
            Mark {
                shape: Shape::Path("M22 22V2"),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Miter,
                }),
            },
            Mark {
                shape: Shape::Path(
                    "M7 18L10.6534 7.48635C10.9447 6.81279 11.4878 5.99657 11.9467 6.00001C12.6263 6.00511 12.9827 6.70758 13.3774 7.48635C13.7721 8.26513 17 18 17 18M9.01312 12.9912L14.8945 12.9293",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
        ],
    },
    IconData {
        name: "subtitles",
        source: Some("SubtitleIcon"),
        marks: &[
            Mark {
                shape: Shape::Path(
                    "M2.5 12C2.5 7.52166 2.5 5.28249 3.89124 3.89124C5.28249 2.5 7.52166 2.5 12 2.5C16.4783 2.5 18.7175 2.5 20.1088 3.89124C21.5 5.28249 21.5 7.52166 21.5 12C21.5 16.4783 21.5 18.7175 20.1088 20.1088C18.7175 21.5 16.4783 21.5 12 21.5C7.52166 21.5 5.28249 21.5 3.89124 20.1088C2.5 18.7175 2.5 16.4783 2.5 12Z",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Butt,
                    join: Join::Miter,
                }),
            },
            Mark {
                shape: Shape::Path("M18 18H10"),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
            Mark {
                shape: Shape::Path("M18 14H15M12 14H10"),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
        ],
    },
    IconData {
        name: "rename",
        source: Some("Edit02Icon"),
        marks: &[
            Mark {
                shape: Shape::Path(
                    "M14.0737 3.88545C14.8189 3.07808 15.1915 2.6744 15.5874 2.43893C16.5427 1.87076 17.7191 1.85309 18.6904 2.39232C19.0929 2.6158 19.4769 3.00812 20.245 3.79276C21.0131 4.5774 21.3972 4.96972 21.6159 5.38093C22.1438 6.37312 22.1265 7.57479 21.5703 8.5507C21.3398 8.95516 20.9446 9.33578 20.1543 10.097L10.7506 19.1543C9.25288 20.5969 8.504 21.3182 7.56806 21.6837C6.63212 22.0493 5.6032 22.0224 3.54536 21.9686L3.26538 21.9613C2.63891 21.9449 2.32567 21.9367 2.14359 21.73C1.9615 21.5234 1.98636 21.2043 2.03608 20.5662L2.06308 20.2197C2.20301 18.4235 2.27297 17.5255 2.62371 16.7182C2.97444 15.9109 3.57944 15.2555 4.78943 13.9445L14.0737 3.88545Z",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Butt,
                    join: Join::Round,
                }),
            },
            Mark {
                shape: Shape::Path("M13 4L20 11"),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Butt,
                    join: Join::Round,
                }),
            },
            Mark {
                shape: Shape::Path("M14 22L22 22"),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
        ],
    },
    IconData {
        name: "settings",
        source: Some("Settings01Icon"),
        marks: &[
            Mark {
                shape: Shape::Path(
                    "M3.95631 10.5829L3.03254 9.07084C2.74505 8.60024 2.7389 8.00978 3.01652 7.5333L3.63533 6.47124C3.90953 6.00064 4.41721 5.71571 4.96162 5.7269L6.69131 5.76245C6.92616 5.76728 7.15379 5.68727 7.34333 5.54846C7.79531 5.21745 8.28334 4.93276 8.80042 4.70139C9.01191 4.60675 9.19222 4.45165 9.30469 4.24903L10.1203 2.77965C10.3847 2.30328 10.8865 2.00781 11.4312 2.00781H12.6937C13.2384 2.00781 13.7403 2.30329 14.0047 2.77966L14.8202 4.24902C14.9327 4.45165 15.113 4.60675 15.3245 4.70139C15.8416 4.93274 16.3296 5.2174 16.7815 5.54838C16.9711 5.68718 17.1987 5.76718 17.4336 5.76235L19.1634 5.72676C19.7078 5.71557 20.2155 6.00049 20.4897 6.47111L21.1085 7.53314C21.3862 8.00963 21.38 8.60009 21.0925 9.07068L20.1687 10.5828C20.0481 10.7803 20.002 11.0134 20.0235 11.2439C20.0707 11.7486 20.0707 12.267 20.0235 12.7717C20.002 13.0022 20.0481 13.2353 20.1687 13.4329L21.0924 14.9448C21.3799 15.4154 21.3861 16.0058 21.1084 16.4823L20.4896 17.5444C20.2154 18.015 19.7077 18.2999 19.1633 18.2887L17.4338 18.2531C17.1989 18.2483 16.9713 18.3283 16.7817 18.4671C16.3297 18.7982 15.8416 19.0829 15.3245 19.3142C15.113 19.4089 14.9327 19.564 14.8202 19.7666L14.0047 21.236C13.7403 21.7123 13.2384 22.0078 12.6937 22.0078H11.4312C10.8866 22.0078 10.3847 21.7123 10.1203 21.236L9.30468 19.7665C9.19222 19.5639 9.01191 19.4088 8.80042 19.3142C8.28329 19.0828 7.79519 18.798 7.34316 18.467C7.15362 18.3282 6.92597 18.2482 6.69111 18.253L4.96175 18.2886C4.41733 18.2998 3.90963 18.0148 3.63543 17.5442L3.01663 16.4822C2.739 16.0057 2.74515 15.4152 3.03264 14.9446L3.95631 13.4327C4.07699 13.2351 4.12309 13.002 4.10154 12.7715C4.05437 12.267 4.05438 11.7486 4.10154 11.244C4.12308 11.0135 4.07699 10.7805 3.95631 10.5829Z",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Butt,
                    join: Join::Miter,
                }),
            },
            Mark {
                shape: Shape::Circle {
                    cx: 12.0625,
                    cy: 12.0078,
                    r: 3.5,
                },
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Butt,
                    join: Join::Miter,
                }),
            },
        ],
    },
    IconData {
        name: "blend",
        source: Some("BlendIcon"),
        marks: &[
            Mark {
                shape: Shape::Circle {
                    cx: 17.0,
                    cy: 17.0,
                    r: 5.0,
                },
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Butt,
                    join: Join::Miter,
                }),
            },
            Mark {
                shape: Shape::Path(
                    "M7.5 13H6.4C4.32582 13 3.28873 13 2.64437 12.3556C2 11.7113 2 10.6742 2 8.6V6.4C2 4.32582 2 3.28873 2.64437 2.64437C3.28873 2 4.32582 2 6.4 2H8.6C10.6742 2 11.7113 2 12.3556 2.64437C13 3.28873 13 4.32582 13 6.4V7.5",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Butt,
                    join: Join::Miter,
                }),
            },
            Mark {
                shape: Shape::Path(
                    "M12.0348 19C9.17734 18.5206 7 16.0355 7 13.0418C7 9.70499 9.70499 7 13.0418 7C16.0355 7 18.5206 9.17734 19 12.0348",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Butt,
                    join: Join::Miter,
                }),
            },
        ],
    },
    IconData {
        name: "mask",
        source: Some("MaskIcon"),
        marks: &[
            Mark {
                shape: Shape::Path(
                    "M6 9C7.46589 8.38677 9.61061 8 12 8C14.3894 8 16.5341 8.38677 18 9",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Miter,
                }),
            },
            Mark {
                shape: Shape::Path(
                    "M9 11.5C9.93974 11.176 10.9492 11 12 11C13.0508 11 14.0603 11.176 15 11.5",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Miter,
                }),
            },
            Mark {
                shape: Shape::Path(
                    "M8 15C9.17669 15.636 10.5429 16 12 16C13.4571 16 14.8233 15.636 16 15",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Miter,
                }),
            },
            Mark {
                shape: Shape::Path(
                    "M6 10C6 13.2571 4.1688 14.9419 3.05731 14.9985C3.02153 15.0003 2.98891 14.9808 2.97139 14.9496C2.15656 13.4966 2 11.7879 2 10C2 6.68629 2.89543 4 4 4C5.10457 4 6 6.68629 6 10Z",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
            Mark {
                shape: Shape::Path(
                    "M18 10C18 13.2571 19.8312 14.9419 20.9427 14.9985C20.9785 15.0003 21.0111 14.9808 21.0286 14.9496C21.8434 13.4966 22 11.7879 22 10C22 6.68629 21.1046 4 20 4C18.8954 4 18 6.68629 18 10Z",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Round,
                }),
            },
            Mark {
                shape: Shape::Path(
                    "M21 15C19.2035 17.989 15.8458 20 12 20C8.15423 20 4.79648 17.989 3 15",
                ),
                fill: false,
                even_odd: false,
                stroke: Some(Stroke {
                    width: 1.5,
                    cap: Cap::Round,
                    join: Join::Miter,
                }),
            },
        ],
    },
    crate::custom::BLADE,
];
