from mmdet3d.apis import inference_detector, init_model, show_result_meshlab




def main():

    pcd = "/data/000011.pcd"
    config = "/mmdetection3d/configs/second/hv_second_secfpn_6x8_80e_kitti-3d-car.py"
    checkpoint = "/data/hv_second_secfpn_6x8_80e_kitti-3d-car_20200620_230238-393f000c.pth"

    score_thr = 0.0

    device = 'cuda:0'
    import open3d as o3d

    # build the model from a config file and a checkpoint file
    model = init_model(config, checkpoint, device=device)
    # test a single image
    result, data = inference_detector(model, pcd)
    # show the results
    print(result, data)







if __name__ == '__main__':
    main()

