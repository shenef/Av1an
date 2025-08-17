#[cfg(test)]
mod tests;

use std::{
    collections::{BTreeMap, HashMap},
    fs::File,
    io::Write,
    path::Path,
    process::{exit, Command},
    str::FromStr,
    sync::atomic,
};

use anyhow::{anyhow, bail, Context, Result};
use itertools::Itertools;
use nom::{
    branch::alt,
    bytes::complete::{tag, take_till, take_while},
    character::complete::{char, digit1, space1},
    combinator::{map, map_res, opt, recognize, rest},
    multi::{many1, separated_list0},
    sequence::preceded,
    Parser,
};
use serde::{Deserialize, Serialize};
use tracing::{info, warn};

use crate::{
    get_done,
    parse::valid_params,
    scene_detect::av_scenechange_detect,
    settings::{invalid_params, suggest_fix},
    split::extra_splits,
    EncodeArgs,
    Encoder,
    SplitMethod,
    TargetMetric,
    TargetQuality,
};

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct Scene {
    pub start_frame:    usize,
    // Reminding again that end_frame is *exclusive*
    pub end_frame:      usize,
    pub zone_overrides: Option<ZoneOptions>,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct ZoneOptions {
    pub encoder:             Encoder,
    pub passes:              u8,
    pub video_params:        Vec<String>,
    pub photon_noise:        Option<u8>,
    pub photon_noise_height: Option<u32>,
    pub photon_noise_width:  Option<u32>,
    pub chroma_noise:        bool,
    pub extra_splits_len:    Option<usize>,
    pub min_scene_len:       usize,
    pub target_quality:      Option<TargetQuality>,
}

impl Scene {
    pub fn parse_from_zone(input: &str, args: &EncodeArgs, frames: usize) -> Result<Self> {
        let (_, (start, _, end, _, encoder, reset, zone_args)): (
            _,
            (usize, _, usize, _, Encoder, bool, &str),
        ) = (
            map_res(digit1::<&str, nom::error::Error<&str>>, str::parse),
            many1(char(' ')),
            map_res(alt((tag("-1"), digit1)), |res: &str| {
                if res == "-1" {
                    Ok(frames)
                } else {
                    res.parse::<usize>()
                }
            }),
            many1(char(' ')),
            map_res(
                alt((
                    tag("aom"),
                    tag("rav1e"),
                    tag("x264"),
                    tag("x265"),
                    tag("vpx"),
                    tag("svt-av1"),
                )),
                Encoder::from_str,
            ),
            map(
                opt(preceded(many1(char(' ')), tag("reset"))),
                |res: Option<&str>| res.is_some(),
            ),
            map(
                opt(preceded(many1(char(' ')), rest)),
                |res: Option<&str>| res.unwrap_or_default().trim(),
            ),
        )
            .parse(input)
            .map_err(|e| anyhow!("Invalid zone file syntax: {}", e))?;
        if start >= end {
            bail!("Start frame must be earlier than the end frame");
        }
        if start >= frames || end > frames {
            bail!("Start and end frames must not be past the end of the video");
        }
        if encoder.format() != args.encoder.format() {
            bail!(
                "Zone specifies using {}, but this cannot be used in the same file as {}",
                encoder,
                args.encoder,
            );
        }
        if encoder != args.encoder {
            if encoder.get_format_bit_depth(args.output_pix_format.format).is_err() {
                bail!(
                    "Output pixel format {:?} is not supported by {} (used in zones file)",
                    args.output_pix_format.format,
                    encoder
                );
            }
            if !reset {
                bail!(
                    "Zone includes encoder change but previous args were kept. You probably meant \
                     to specify \"reset\"."
                );
            }
        }

        // Inherit from encode args or reset to defaults
        let mut video_params = if reset {
            Vec::new()
        } else {
            args.video_params.clone()
        };
        let mut passes = if reset {
            encoder.get_default_pass()
        } else {
            args.passes
        };
        let mut photon_noise = if reset { None } else { args.photon_noise };
        let mut photon_noise_height = if reset {
            None
        } else {
            args.photon_noise_size.1
        };
        let mut photon_noise_width = if reset {
            None
        } else {
            args.photon_noise_size.0
        };
        let mut chroma_noise = if reset { false } else { args.chroma_noise };
        let mut extra_splits_len = args.extra_splits_len;
        let mut min_scene_len = args.min_scene_len;

        // Target Quality options
        let mut target_quality = args.target_quality.clone();

        // Parse overrides
        let zone_args: (&str, Vec<(&str, Option<&str>)>) =
            separated_list0::<_, nom::error::Error<&str>, _, _>(
                space1,
                (
                    recognize((
                        alt((tag("--"), tag("-"))),
                        take_till(|c| c == '=' || c == ' '),
                    )),
                    opt(preceded(alt((space1, tag("="))), take_while(|c| c != ' '))),
                ),
            )
            .parse(zone_args)
            .map_err(|e| anyhow!("Invalid zone file syntax: {}", e))?;
        let mut zone_args = zone_args.1.into_iter().collect::<HashMap<_, _>>();
        if let Some(Some(zone_passes)) = zone_args.remove("--passes") {
            passes = zone_passes.parse()?;
        } else if [Encoder::aom, Encoder::vpx].contains(&encoder) && zone_args.contains_key("--rt")
        {
            passes = 1;
        }
        if let Some(Some(zone_photon_noise)) = zone_args.remove("--photon-noise") {
            photon_noise = Some(zone_photon_noise.parse()?);
        }
        if let Some(Some(zone_photon_noise_height)) = zone_args.remove("--photon-noise-height") {
            photon_noise_height = Some(zone_photon_noise_height.parse()?);
        }
        if let Some(Some(zone_photon_noise_width)) = zone_args.remove("--photon-noise-width") {
            photon_noise_width = Some(zone_photon_noise_width.parse()?);
        }
        if let Some(Some(zone_chroma_noise)) = zone_args.remove("--chroma-noise") {
            chroma_noise = zone_chroma_noise.parse()?;
        }
        if let Some(Some(zone_xs)) =
            zone_args.remove("-x").or_else(|| zone_args.remove("--extra-split"))
        {
            extra_splits_len = Some(zone_xs.parse()?);
        }
        if let Some(Some(zone_min_scene_len)) = zone_args.remove("--min-scene-len") {
            min_scene_len = zone_min_scene_len.parse()?;
        }
        if let Some(Some(zone_target_quality)) = zone_args.remove("--target-quality") {
            let parsed = TargetQuality::parse_target_qp_range(zone_target_quality)
                .map_err(|e| anyhow!("Invalid --target-quality: {}", e))?;
            target_quality.target = Some(parsed);
        }
        if let Some(Some(zone_target_metric)) = zone_args.remove("--target-metric") {
            let parsed = TargetMetric::from_str(zone_target_metric)
                .map_err(|_| anyhow!("Invalid --target-metric: {}", zone_target_metric))?;
            target_quality.metric = parsed;
        }
        if let Some(Some(zone_qp_range)) = zone_args.remove("--qp-range") {
            let (min, max) = TargetQuality::parse_qp_range(zone_qp_range)
                .map_err(|e| anyhow!("Invalid --qp-range: {}", e))?;
            target_quality.min_q = min;
            target_quality.max_q = max;
        }
        if let Some(Some(zone_probes)) = zone_args.remove("--probes") {
            let parsed =
                zone_probes.parse().map_err(|_| anyhow!("Invalid --probes: {}", zone_probes))?;
            let (probes, warning) = TargetQuality::validate_probes(parsed)
                .map_err(|e| anyhow!("Invalid --probes: {}: {}", parsed, e))?;
            if let Some(warning) = warning {
                warn!("{}", warning);
            }
            target_quality.probes = probes;
        }
        if let Some(Some(zone_probing_rate)) = zone_args.remove("--probing-rate") {
            let parsed = zone_probing_rate
                .parse()
                .map_err(|_| anyhow!("Invalid --probing-rate: {}", zone_probing_rate))?;
            let (probing_rate, warning) = TargetQuality::validate_probing_rate(parsed)
                .map_err(|e| anyhow!("Invalid --probing-rate: {}: {}", parsed, e))?;
            if let Some(warning) = warning {
                warn!("{}", warning);
            }
            target_quality.probing_rate = probing_rate;
        }
        if let Some(Some(zone_probe_res)) = zone_args.remove("--probe-res") {
            let (width, height) = TargetQuality::parse_probe_res(zone_probe_res)
                .map_err(|e| anyhow!("Invalid --probe-res: {}", e))?;
            target_quality.probe_res = Some((width, height));
        }
        if let Some(Some(zone_probing_stat)) = zone_args.remove("--probing-stat") {
            let parsed = TargetQuality::parse_probing_statistic(zone_probing_stat)
                .map_err(|_| anyhow!("Invalid --probing-stat: {}", zone_probing_stat))?;
            target_quality.probing_statistic = parsed;
        }
        if let Some(Some(zone_interp_method)) = zone_args.remove("--interp-method") {
            let (method4, method5) = TargetQuality::parse_interp_method(zone_interp_method)
                .map_err(|e| anyhow!("Invalid --interp-method: {}", e))?;
            target_quality.interp_method = Some((method4, method5));
        }

        let raw_zone_args = if [Encoder::aom, Encoder::vpx].contains(&encoder) {
            zone_args
                .into_iter()
                .map(|(key, value)| {
                    value.map_or_else(|| key.to_string(), |value| format!("{key}={value}"))
                })
                .collect::<Vec<String>>()
        } else {
            zone_args
                .keys()
                .map(|&k| Some(k.to_string()))
                .interleave(zone_args.values().map(|v| v.map(std::string::ToString::to_string)))
                .flatten()
                .collect::<Vec<String>>()
        };

        if !args.force {
            let help_text = {
                let [cmd, arg] = encoder.help_command();
                String::from_utf8_lossy(&Command::new(cmd).arg(arg).output()?.stdout).to_string()
            };
            let valid_params = valid_params(&help_text, encoder);
            let interleaved_args: Vec<&str> = raw_zone_args
                .iter()
                .filter_map(|param| {
                    if param.starts_with('-') && [Encoder::aom, Encoder::vpx].contains(&encoder) {
                        // These encoders require args to be passed using an equal sign,
                        // e.g. `--cq-level=30`
                        param.split('=').next()
                    } else {
                        // The other encoders use a space, so we don't need to do extra splitting,
                        // e.g. `--crf 30`
                        None
                    }
                })
                .collect();
            let invalid_params = invalid_params(&interleaved_args, &valid_params);

            for wrong_param in &invalid_params {
                eprintln!("'{wrong_param}' isn't a valid parameter for {encoder}");
                if let Some(suggestion) = suggest_fix(wrong_param, &valid_params) {
                    eprintln!("\tDid you mean '{suggestion}'?");
                }
            }

            if !invalid_params.is_empty() {
                println!("\nTo continue anyway, run av1an with '--force'");
                exit(1);
            }
        }

        for arg in raw_zone_args {
            if arg.starts_with("--")
                || (arg.starts_with('-') && arg.chars().nth(1).is_some_and(char::is_alphabetic))
            {
                let key = arg.split_once('=').map_or(arg.as_str(), |split| split.0);
                if let Some(pos) = video_params
                    .iter()
                    .position(|param| param == key || param.starts_with(&format!("{key}=")))
                {
                    video_params.remove(pos);
                    if let Some(next) = video_params.get(pos) {
                        if !([Encoder::aom, Encoder::vpx].contains(&encoder)
                            || next.starts_with("--")
                            || (next.starts_with('-')
                                && next.chars().nth(1).is_some_and(char::is_alphabetic)))
                        {
                            video_params.remove(pos);
                        }
                    }
                }
            }
            video_params.push(arg);
        }

        Ok(Self {
            start_frame:    start,
            end_frame:      end,
            zone_overrides: Some(ZoneOptions {
                encoder,
                passes,
                video_params,
                photon_noise,
                photon_noise_height,
                photon_noise_width,
                chroma_noise,
                extra_splits_len,
                min_scene_len,
                target_quality: Some(target_quality),
            }),
        })
    }
}

/// This struct is responsible for choosing and building a list of video chunks.
/// It is responsible for managing both scene detection and extra splits.
#[derive(Debug)]
pub struct SceneFactory {
    data: ScenesData,
}

/// A serializable data struct containing scenecut data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScenesData {
    frames:       usize,
    scenes:       Option<Vec<Scene>>,
    split_scenes: Option<Vec<Scene>>,
}

impl SceneFactory {
    /// Return a new, empty factory for computing scenes and chunks.
    pub fn new() -> Self {
        Self {
            data: ScenesData {
                frames:       0,
                scenes:       None,
                split_scenes: None,
            },
        }
    }

    /// This loads a list of scenes from a JSON file and returns a factory with
    /// the scenes data.
    pub fn from_scenes_file<P: AsRef<Path>>(scene_path: &P) -> anyhow::Result<Self> {
        let file = File::open(scene_path)?;
        let mut data: ScenesData = serde_json::from_reader(file).with_context(|| {
            format!(
                "Failed to parse scenes file {}, this likely means that the scenes file is \
                 corrupted",
                scene_path.as_ref().display()
            )
        })?;
        if data.scenes.is_some() && data.split_scenes.is_none() {
            // If a user is using a scenes file from an older build of av1an,
            // we need to copy the list of scenes to the `split_scenes` array.
            // We won't be able to do special pre-split analysis because the scenes
            // list in the old format was always post-split, but at least the
            // encode won't error out.
            data.split_scenes = data.scenes.clone();
        }
        get_done().frames.store(data.frames, atomic::Ordering::SeqCst);

        Ok(Self {
            data,
        })
    }

    /// Retrieve the pre-extra-split scenes data
    #[expect(dead_code)]
    pub fn get_scenecuts(&self) -> anyhow::Result<&[Scene]> {
        if self.data.scenes.is_none() {
            bail!("compute_scenes must be called first");
        }

        Ok(self.data.scenes.as_deref().expect("scenes exist"))
    }

    /// Retrieve the post-extra-split scenes data
    pub fn get_split_scenes(&self) -> anyhow::Result<&[Scene]> {
        if self.data.split_scenes.is_none() {
            bail!("compute_scenes must be called first");
        }

        Ok(self.data.split_scenes.as_deref().expect("split_scenes exist"))
    }

    pub fn get_frame_count(&self) -> usize {
        self.data.frames
    }

    /// Write the scenes data to the specified file as JSON
    pub fn write_scenes_to_file<P: AsRef<Path>>(&self, scene_path: P) -> anyhow::Result<()> {
        if self.data.scenes.is_none() {
            bail!("compute_scenes must be called first");
        }

        let json = serde_json::to_string_pretty(&self.data).expect("serialize should not fail");

        let mut file = File::create(scene_path)?;
        file.write_all(json.as_bytes())?;

        Ok(())
    }

    /// This runs scene detection and populates a list of scenes into the
    /// factory. This function must be called before getting the list of scenes
    /// or writing to the file.
    pub fn compute_scenes(&mut self, args: &EncodeArgs, zones: &[Scene]) -> anyhow::Result<()> {
        // We should only be calling this when scenes haven't been created yet
        debug_assert!(self.data.scenes.is_none());

        let frames = args.input.clip_info()?.num_frames;

        let (mut scenes, frames, scores) = match args.split_method {
            SplitMethod::AvScenechange => av_scenechange_detect(
                args.proxy.as_ref().unwrap_or(&args.input),
                args.encoder,
                frames,
                args.min_scene_len,
                args.verbosity,
                args.scaler.as_str(),
                args.sc_pix_format,
                args.sc_method,
                args.sc_downscale_height,
                zones,
            )?,
            SplitMethod::None => {
                let mut scenes = Vec::with_capacity(2 * zones.len() + 1);
                let mut frames_processed = 0;
                // Add scenes for each zone and the scenes between zones
                for zone in zones {
                    // Frames between the previous zone and this zone
                    if zone.start_frame > frames_processed {
                        // No overrides for unspecified frames between zones
                        scenes.push(Scene {
                            start_frame:    frames_processed,
                            end_frame:      zone.start_frame,
                            zone_overrides: None,
                        });
                    }

                    // Add the zone with its overrides
                    scenes.push(zone.clone());
                    // Update the frames processed
                    frames_processed = zone.end_frame;
                }
                if frames > frames_processed {
                    scenes.push(Scene {
                        start_frame:    frames_processed,
                        end_frame:      frames,
                        zone_overrides: None,
                    });
                }
                (scenes, frames, BTreeMap::new())
            },
        };

        self.data.frames = frames;
        get_done().frames.store(frames, atomic::Ordering::SeqCst);

        // Add forced keyframes
        for kf in &args.force_keyframes {
            if let Some((scene_pos, s)) =
                scenes.iter_mut().find_position(|s| (s.start_frame..s.end_frame).contains(kf))
            {
                if *kf == s.start_frame {
                    // Already a keyframe
                    continue;
                }
                // Split this scene into two scenes at the requested keyframe
                let mut new = s.clone();
                s.end_frame = *kf;
                new.start_frame = *kf;
                scenes.insert(scene_pos + 1, new);
            } else {
                warn!(
                    "scene {kf} was requested as a forced keyframe but video has {frames} frames, \
                     ignoring"
                );
            }
        }

        if let Some(scene) = scenes.last() {
            assert!(
                scene.end_frame <= frames,
                "scenecut reported at frame {}, but there are only {} frames",
                scene.end_frame,
                frames
            );
        }

        let scenes_before = scenes.len();
        self.data.scenes = Some(scenes);

        if let Some(split_len @ 1..) = args.extra_splits_len {
            self.data.split_scenes = Some(extra_splits(
                self.data.scenes.as_deref().expect("scenes is set"),
                split_len,
                &scores,
            ));
            let scenes_after = self.data.split_scenes.as_ref().expect("split_scenes is set").len();
            info!(
                "scenecut: found {scenes_before} scene(s) [with extra_splits ({split_len} \
                 frames): {scenes_after} scene(s)]"
            );
        } else {
            self.data.split_scenes = self.data.scenes.clone();
            info!("scenecut: found {scenes_before} scene(s)");
        }

        Ok(())
    }
}
