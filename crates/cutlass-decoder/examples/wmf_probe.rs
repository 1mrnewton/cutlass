//! Diagnostic: how does `MF_PD_DURATION` look through each source-reader
//! open path on this machine, and what does the MP4 container itself claim?
//!
//! Usage: `cargo run --release -p cutlass-decoder --example wmf_probe -- <file>`

#[cfg(target_os = "windows")]
fn main() {
    use std::path::Path;

    use windows::Win32::Media::MediaFoundation::{
        IMFMediaSource, IMFSourceReader, MF_ACCESSMODE_READ, MF_FILEFLAGS_NONE,
        MF_OPENMODE_FAIL_IF_NOT_EXIST, MF_PD_DURATION, MF_SOURCE_READER_FIRST_VIDEO_STREAM,
        MF_SOURCE_READER_MEDIASOURCE, MF_VERSION, MFCreateFile, MFCreateSourceReaderFromByteStream,
        MFCreateSourceReaderFromURL, MFSTARTUP_FULL, MFStartup,
    };
    use windows::Win32::System::Com::{COINIT_MULTITHREADED, CoInitializeEx};
    use windows::core::{GUID, HSTRING, Interface};

    fn dump_duration(reader: &IMFSourceReader, label: &str) {
        let selectors = [
            ("MEDIASOURCE", MF_SOURCE_READER_MEDIASOURCE.0 as u32),
            (
                "FIRST_VIDEO_STREAM",
                MF_SOURCE_READER_FIRST_VIDEO_STREAM.0 as u32,
            ),
        ];
        for (name, index) in selectors {
            match unsafe { reader.GetPresentationAttribute(index, &MF_PD_DURATION) } {
                Ok(value) => {
                    println!("[{label}] GetPresentationAttribute({name}) -> Ok");
                    println!("    debug   : {value:?}");
                    println!("    u64     : {:?}", u64::try_from(&value));
                }
                Err(e) => {
                    println!("[{label}] GetPresentationAttribute({name}) -> Err {e:?}");
                }
            }
        }
    }

    /// Dump every attribute on the media source's presentation descriptor.
    fn dump_presentation_descriptor(reader: &IMFSourceReader) {
        let source: Result<IMFMediaSource, _> = unsafe {
            let mut ptr = core::ptr::null_mut();
            reader
                .GetServiceForStream(
                    MF_SOURCE_READER_MEDIASOURCE.0 as u32,
                    &GUID::zeroed(),
                    &IMFMediaSource::IID,
                    &mut ptr,
                )
                .map(|()| IMFMediaSource::from_raw(ptr))
        };
        let source = match source {
            Ok(s) => s,
            Err(e) => {
                println!("GetServiceForStream(media source) failed: {e:?}");
                return;
            }
        };
        let pd = match unsafe { source.CreatePresentationDescriptor() } {
            Ok(pd) => pd,
            Err(e) => {
                println!("CreatePresentationDescriptor failed: {e:?}");
                return;
            }
        };
        let count = unsafe { pd.GetCount() }.unwrap_or(0);
        println!("presentation descriptor: {count} attributes");
        for i in 0..count {
            let mut guid = GUID::zeroed();
            match unsafe { pd.GetItemByIndex(i, &mut guid, None) } {
                Ok(()) => {
                    let known = if guid == MF_PD_DURATION {
                        " (MF_PD_DURATION)"
                    } else {
                        ""
                    };
                    println!("  [{i}] {guid:?}{known}");
                }
                Err(e) => println!("  [{i}] GetItemByIndex failed: {e:?}"),
            }
        }
        match unsafe { pd.GetUINT64(&MF_PD_DURATION) } {
            Ok(v) => println!("pd.GetUINT64(MF_PD_DURATION) = {v}"),
            Err(e) => println!("pd.GetUINT64(MF_PD_DURATION) -> Err {e:?}"),
        }
    }

    // ---- raw MP4 box scan (ground truth on what the container claims) ----

    fn be32(b: &[u8]) -> u64 {
        u64::from(u32::from_be_bytes([b[0], b[1], b[2], b[3]]))
    }
    fn be64(b: &[u8]) -> u64 {
        u64::from_be_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]])
    }

    /// Walk `data` as a sequence of MP4 boxes, printing timing-relevant ones.
    fn walk_boxes(data: &[u8], depth: usize, path: &str) {
        let mut off = 0usize;
        while off + 8 <= data.len() {
            let size = be32(&data[off..]) as usize;
            let kind = String::from_utf8_lossy(&data[off + 4..off + 8]).into_owned();
            let (header, size) = if size == 1 {
                if off + 16 > data.len() {
                    return;
                }
                (16usize, be64(&data[off + 8..]) as usize)
            } else if size == 0 {
                (8usize, data.len() - off)
            } else {
                (8usize, size)
            };
            if size < header || off + size > data.len() {
                println!(
                    "{:indent$}{path}/{kind}: malformed size {size}",
                    "",
                    indent = depth * 2
                );
                return;
            }
            let body = &data[off + header..off + size];
            match kind.as_str() {
                "moov" | "trak" | "mdia" | "mvex" => {
                    println!("{:indent$}{kind} ({size} bytes)", "", indent = depth * 2);
                    walk_boxes(body, depth + 1, &format!("{path}/{kind}"));
                }
                "mvhd" | "mdhd" => {
                    let version = body[0];
                    let (timescale, duration) = if version == 1 {
                        (be32(&body[20..]), be64(&body[24..]))
                    } else {
                        (be32(&body[12..]), be32(&body[16..]))
                    };
                    let secs = if timescale > 0 {
                        duration as f64 / timescale as f64
                    } else {
                        f64::NAN
                    };
                    println!(
                        "{:indent$}{kind} v{version}: timescale={timescale} duration={duration} (~{secs:.3}s)",
                        "",
                        indent = depth * 2
                    );
                }
                "tkhd" => {
                    let version = body[0];
                    let duration = if version == 1 {
                        be64(&body[28..])
                    } else {
                        be32(&body[20..])
                    };
                    println!(
                        "{:indent$}tkhd v{version}: duration={duration} (movie timescale)",
                        "",
                        indent = depth * 2
                    );
                }
                "mehd" => {
                    let version = body[0];
                    let duration = if version == 1 {
                        be64(&body[4..])
                    } else {
                        be32(&body[4..])
                    };
                    println!(
                        "{:indent$}mehd v{version}: fragment_duration={duration}",
                        "",
                        indent = depth * 2
                    );
                }
                _ => {
                    if depth == 0 {
                        println!("{kind} ({size} bytes)");
                    }
                }
            }
            off += size;
        }
    }

    let path = std::env::args().nth(1).unwrap_or_else(|| {
        r"C:\Users\Mr. Newton\Downloads\16265742_3840_2160_30fps.mp4".to_string()
    });
    println!("file: {path}");

    println!("\n== raw MP4 boxes ==");
    match std::fs::read(&path) {
        Ok(data) => walk_boxes(&data, 0, ""),
        Err(e) => println!("read failed: {e}"),
    }

    unsafe {
        let _ = CoInitializeEx(None, COINIT_MULTITHREADED);
        MFStartup(MF_VERSION, MFSTARTUP_FULL).expect("MFStartup");
    }

    let url = HSTRING::from(path.as_str());

    println!("\n== byte-stream reader ==");
    let byte_stream_reader = unsafe {
        MFCreateFile(
            MF_ACCESSMODE_READ,
            MF_OPENMODE_FAIL_IF_NOT_EXIST,
            MF_FILEFLAGS_NONE,
            &url,
        )
    }
    .and_then(|stream| unsafe { MFCreateSourceReaderFromByteStream(&stream, None) });
    match &byte_stream_reader {
        Ok(reader) => {
            dump_duration(reader, "bytestream");
            dump_presentation_descriptor(reader);
        }
        Err(e) => println!("byte-stream open failed: {e:?}"),
    }

    println!("\n== URL reader ==");
    match unsafe { MFCreateSourceReaderFromURL(&url, None) } {
        Ok(reader) => {
            dump_duration(&reader, "url");
            dump_presentation_descriptor(&reader);
        }
        Err(e) => println!("URL open failed: {e:?}"),
    }

    println!("\n== cutlass_decoder::probe ==");
    match cutlass_decoder::probe(Path::new(&path)) {
        Ok(probe) => println!("{probe:#?}"),
        Err(e) => println!("probe failed: {e:?}"),
    }

    println!("\n== decoder open ==");
    match cutlass_decoder::WmfDecoder::open(Path::new(&path), cutlass_decoder::OutputMode::Cpu) {
        Ok(decoder) => println!(
            "hardware accelerated: {}",
            decoder.is_hardware_accelerated()
        ),
        Err(e) => println!("open failed: {e}"),
    }
}

#[cfg(not(target_os = "windows"))]
fn main() {
    eprintln!("wmf_probe is Windows-only");
}
