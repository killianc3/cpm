use image::{io, GenericImageView, Luma};
use linfa::prelude::*;

use std::f32::consts::PI;

const CELL: (u32, u32) = (8, 8);
const BLOCK: (u32, u32) = (CELL.0 * 2, CELL.1 * 2);
const STRIDE: (u32, u32) = (CELL.0, CELL.1);
const DIMS: u32 = 9;

fn main() {
    let mut data = ndarray::Array2::<f32>::zeros((0, 1764));
    let mut targets = vec![];

    for (is_vehicle, dir) in ["data/vehicles", "data/non-vehicles"].iter().enumerate() {
    for (c, entry) in std::fs::read_dir(dir).unwrap().enumerate() {
        println!("{c} {dir}");
        let mut features = vec![];

        let img = image::io::Reader::open(entry.unwrap().path()).unwrap().decode().unwrap();
        let gray_img = img.into_luma8();

        let mut magnitudes = [0.0; 64 * 64];
        let mut directions = [0.0; 64 * 64];
        for i in 0..64 {
            for j in 0..64 {
                let gx = gray_img.get_pixel(if i == 0 { i } else { i - 1 }, j).0[0] as f32 - gray_img.get_pixel(if i == 63 { i } else { i + 1 }, j).0[0] as f32;
                let gy = gray_img.get_pixel(i, if j == 0 { j } else { j - 1 }).0[0] as f32 - gray_img.get_pixel(i, if j == 63 { j } else { j + 1 }).0[0] as f32;

                magnitudes[(j * 64 + i) as usize] = (gx * gx + gy * gy).sqrt();
                directions[(j * 64 + i) as usize] = (gy/gx).atan();
            }
        }

        for bi in (0..(64 - 8)).step_by(8) {
            for bj in (0..(64 - 8)).step_by(8) {
                let mut block_features = [0.0; 36];
                let mut block_offset = 0;

                for ci in (bi..(bi + 16)).step_by(8) {
                    for cj in (bj..(bj + 16)).step_by(8) {
                        let mut magnitudes_sum = 0.0;

                        for i in ci..(ci + 8) {
                            for j in cj..(cj + 8) {
                                magnitudes_sum += magnitudes[j * 64 + i];
                            }
                        }

                        for i in ci..(ci + 8) {
                            for j in cj..(cj + 8) {
                                if magnitudes_sum > 0.0 {
                                    block_features[block_offset * 9 + (((directions[j * 64 + i] + (PI / 2.0)) / PI) * 8.0) as usize] += magnitudes[j * 64 + i] / magnitudes_sum;
                                }
                            }
                        }
                        block_offset += 1;
                    }
                }

                let mut l2_norm = 0.0;
                for feature in &block_features {
                    l2_norm += feature * feature;
                }
                l2_norm = l2_norm.sqrt();
                
                for feature in block_features {
                    features.push(if feature.is_normal() { feature / l2_norm } else { 0.0 });
                }
            }
        }

        data.push_row(ndarray::ArrayView::from(features.as_slice())).unwrap();
        targets.push(is_vehicle == 0);
    }
    }


    let data = linfa::DatasetBase::new(data, ndarray::Array::from_vec(targets));
    println!("{} {}", data.nsamples(), data.nfeatures());
    let mut aaaa = rand::thread_rng();
    
    let data = data.shuffle(&mut aaaa);
    let (data, _) = data.split_with_ratio(0.5);
    let (train, valid) = data.split_with_ratio(0.8);
    let svm = linfa_svm::Svm::<_, bool>::params().linear_kernel().fit(&train).unwrap();
    println!("{}", svm);

     let pred = svm.predict(&valid);

    // create a confusion matrix
    let cm = pred.confusion_matrix(&valid).unwrap();

    // Print the confusion matrix, this will print a table with four entries. On the diagonal are
    // the number of true-positive and true-negative predictions, off the diagonal are
    // false-positive and false-negative
    println!("{:?}", cm);

    // Calculate the accuracy and Matthew Correlation Coefficient (cross-correlation between
    // predicted and targets)
    println!("accuracy {}, MCC {}", cm.accuracy(), cm.mcc());

    /*
    let img = io::Reader::open("image.jpg").unwrap().decode().unwrap();
    let gray_img = img.into_luma8();
    let nb_blocks = ((gray_img.width() - 2 - (BLOCK.0 - STRIDE.0)) / STRIDE.0, (gray_img.height() - 2 - (BLOCK.1 - STRIDE.1)) / STRIDE.1);

    let mut magnitudes: Vec<f32> = Vec::with_capacity(((nb_blocks.0 * STRIDE.0 + (BLOCK.0 - STRIDE.0)) * (nb_blocks.1 * STRIDE.1 + (BLOCK.1 - STRIDE.1))) as usize);
    let mut directions: Vec<f32> = Vec::with_capacity(magnitudes.capacity());
    let mut features: Vec<f32> = Vec::with_capacity((nb_blocks.0 * nb_blocks.1 * (BLOCK.0 / CELL.0) * (BLOCK.1 / CELL.1) * DIMS) as usize);

    for i in 0..(nb_blocks.0 * STRIDE.0 + (BLOCK.0 - STRIDE.0)) {
        for j in 0..(nb_blocks.1 * STRIDE.1 + (BLOCK.1 - STRIDE.1)) {
            let gx = gray_img.get_pixel(i, j + 1).0[0] as f32 - gray_img.get_pixel(i + 2, j + 1).0[0] as f32;
            let gy = gray_img.get_pixel(i + 1, j).0[0] as f32 - gray_img.get_pixel(i + 1, j + 2).0[0] as f32;

            magnitudes.push((gx * gx + gy * gy).sqrt());
            directions.push((gy/gx).atan());
        }
    }

    for bi in (0..(nb_blocks.0 * STRIDE.0 + (BLOCK.0 - STRIDE.0)) - STRIDE.0).step_by(STRIDE.0 as usize) {
        for bj in (0..(nb_blocks.1 * STRIDE.1 + (BLOCK.1 - STRIDE.1)) - STRIDE.1).step_by(STRIDE.1 as usize) {
            //println!("block({}/{}, {}/{})", bi / STRIDE.0, nb_blocks.0 - 1, bj / STRIDE.1, nb_blocks.1 - 1);
            let mut block_features = [0.0; (DIMS * (BLOCK.0 / CELL.0) * (BLOCK.1 / CELL.1)) as usize];
            let mut block_of = 0;
            
            for ci in (bi..(bi + BLOCK.0)).step_by(CELL.0 as usize) {
                for cj in (bj..(bj + BLOCK.1)).step_by(CELL.1 as usize) {
                    let mut magnitude_sum = 0.0;

                    for i in ci..(ci + CELL.0) {
                        for j in cj..(cj + CELL.1) {
                            magnitude_sum += magnitudes[(j * (nb_blocks.0 * STRIDE.0 + (BLOCK.0 - STRIDE.0)) + i) as usize];
                        }
                    }

                    for i in ci..(ci + CELL.0) {
                        for j in cj..(cj + CELL.1) {
                            let magnitude = magnitudes[(j * (nb_blocks.0 * STRIDE.0 + (BLOCK.0 - STRIDE.0)) + i) as usize];
                            let direction = directions[(j * (nb_blocks.0 * STRIDE.0 + (BLOCK.0 - STRIDE.0)) + i) as usize];
                            block_features[DIMS as usize * block_of + (((direction + (PI / 2.0)) / PI) * (DIMS - 1) as f32) as usize] += magnitude / magnitude_sum;
                        }
                    }
                    block_of += 1;
                }
            }

            let mut l2_norm = 0.0;
            for feature in &block_features {
                l2_norm += feature * feature;
            }
            l2_norm = l2_norm.sqrt();

            for feature in block_features {
                features.push(feature / l2_norm);
            }
        }
    }

    //println!("{} {}", features.len(), features.capacity());
    //println!("{:?}", features);

    let data = linfa::DatasetBase::new(ndarray::Array::from_shape_vec((4, features.len() / 4), features).unwrap(), ndarray::array!(false, false, false, true));
    let svm = linfa_svm::Svm::<_, bool>::params().gaussian_kernel(10.0).fit(&data).unwrap();
    println!("{}", svm);
    */
}
