#from losses
if mask_dyn:
    mask_dyn_fw = tf.sigmoid(mask_dyn_fw)
    mask_dyn_bw = tf.sigmoid(mask_dyn_bw)
    losses['mask_dyn'] = (charbonnier_loss(1 - mask_dyn_fw) + charbonnier_loss(1 - mask_dyn_bw))
        if mask_dyn:
            losses['epipolar'] = (epipolar_loss(flow_fw, pose_fw, intrinsic, mask_fw * mask_dyn_fw) + \
                                 epipolar_loss(flow_bw, pose_bw, intrinsic, mask_bw * mask_dyn_bw))


def sym_pose_loss(pose_fw, pose_bw):
    batch_size, _ = tf.unstack(tf.shape(pose_fw))
    pose_mul = tf.matmul(pose_vec2mat(pose_fw), pose_vec2mat(pose_bw))
    identity_mat = tf.eye(4, batch_shape=[4])
    sym_pose_error = tf.pow(tf.square(pose_mul - identity_mat) + tf.square(0.001), 0.45)
    return tf.reduce_mean(sym_pose_error)

# from unsupervised
mask_dyn = False
masks_dyn_fw = []
masks_dyn_bw = []
if mask_dyn:
    flows_fw, flows_bw, poses_fw, poses_bw, masks_dyn_fw, masks_dyn_bw = flownet(im1_photo, im2_photo,
                                                                                 flownet_spec=flownet_spec,
                                                                                 full_resolution=full_resolution,
                                                                                 backward_flow=True,
                                                                                 mask_dyn=True,
                                                                                 train_all=train_all)))
    if mask_dyn:
        masks_dyn_fw = masks_dyn_fw[-1]
        masks_dyn_bw = masks_dyn_bw[-1]
        tf.add_to_collection('train_images', tf.identity(masks_dyn_fw[0], name='mask_fw'))
    if mask_dyn:
        losses = compute_losses(im1_s, im2_s,
                                flow_fw_s * flow_scale, flow_bw_s * flow_scale,
                                poses_fw, [i]
    poses_bw, intrinsics[:, i + 2, :, :],
    masks_dyn_fw[i], masks_dyn_bw[i], mask_dyn = True,
                                                 border_mask = mask_s if params.get('border_mask') else None,
                                                               mask_occlusion = mask_occlusion,
                                                                                data_max_distance =
    layer_patch_distances[i])
                if mask_dyn:
                    losses = compute_losses(im1_s, im2_s,
                                        flow_fw_s * flow_scale, flow_bw_s * flow_scale,
                                        mask_dyn_fw=masks_dyn_fw[i], mask_dyn_bw=masks_dyn_bw[i], mask_dyn=True,
                                        border_mask=mask_s if params.get('border_mask') else None,
                                        mask_occlusion=mask_occlusion,
                                        data_max_distance=layer_patch_distances[i])

_, _, h, w = tf.unstack(tf.shape(conv1))
conv1_shape = tf.stack([1, 1, h, w])
conv1 = tf.concat([conv1, tf.tile(pose_avg, conv1_shape)], axis=1)
_, _, h, w = tf.unstack(tf.shape(conv2))
conv2_shape = tf.stack([1, 1, h, w])
conv2 = tf.concat([conv2, tf.tile(pose_avg, conv2_shape)], axis=1)
_, _, h, w = tf.unstack(tf.shape(conv3_1))
conv3_shape = tf.stack([1, 1, h, w])
conv3_1 = tf.concat([conv3_1, tf.tile(pose_avg, conv3_shape)], axis=1)
_, _, h, w = tf.unstack(tf.shape(conv4_1))
conv4_shape = tf.stack([1, 1, h, w])
conv4_1 = tf.concat([conv4_1, tf.tile(pose_avg, conv4_shape)], axis=1)
_, _, h, w = tf.unstack(tf.shape(conv5_1))
conv5_shape = tf.stack([1, 1, h, w])
conv5_1 = tf.concat([conv5_1, tf.tile(pose_avg, conv5_shape)], axis=1)
_, _, h, w = tf.unstack(tf.shape(conv6_1))
conv6_shape = tf.stack([1, 1, h, w])
conv6_1 = tf.concat([conv6_1, tf.tile(pose_avg, conv6_shape)], axis=1)

conv1 = slim.conv2d(inputs, int(64 * m), 7, stride=2, scope='conv1')
        conv2 = slim.conv2d(conv1, int(128 * m), 5, stride=2, scope='conv2')
        conv3 = slim.conv2d(conv2, int(256 * m), 5, stride=2, scope='conv3')
        conv3_1 = slim.conv2d(conv3, int(256 * m), 3, stride=1, scope='conv3_1')
        conv4 = slim.conv2d(conv3_1, int(512 * m), 3, stride=2, scope='conv4')
        conv4_1 = slim.conv2d(conv4, int(512 * m), 3, stride=1, scope='conv4_1')
        conv5 = slim.conv2d(conv4_1, int(512 * m), 3, stride=2, scope='conv5')
        conv5_1 = slim.conv2d(conv5, int(512 * m), 3, stride=1, scope='conv5_1')
        conv6 = slim.conv2d(conv5_1, int(1024 * m), 3, stride=2, scope='conv6')
        conv6_1 = slim.conv2d(conv6, int(1024 * m), 3, stride=1, scope='conv6_1')
        conv7  = slim.conv2d(conv6_1, int(1024 * m), 3, stride=1, scope='conv7')
        conv7_1  = slim.conv2d(conv7, num_dyn, 3, stride=1, scope='conv7_1')

def _mask_upconv(inputs, num_dyn=3, full_res=False, reuse=False):
    with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                        reuse=reuse):
        _, H, W, _ = tf.unstack(tf.shape(inputs[0]))
        if full_res:
            icnv0 = slim.conv2d(inputs[6], 8, 7, scope='icnv0', activation_fn=None)
            mask0 = slim.conv2d(icnv0, num_dyn, 3, scope='mask0', activation_fn=None)
            mask0d = tf.image.resize_bilinear(mask0, [H * 32, W * 32])
            dcnv1 = slim.conv2d(icnv0, 8, 3, stride=2, scope='conv1')

            mask1_in = tf.concat([mask0d, dcnv1, inputs[5]], axis=3)
            icnv1 = slim.conv2d(mask1_in, 8, 7, scope='icnv1', activation_fn=None)
            mask1 = slim.conv2d(icnv1, num_dyn, 3, scope='mask1', activation_fn=None)
            mask1d = tf.image.resize_bilinear(mask1, [H * 16, W * 16])
            dcnv2 = slim.conv2d(mask1d, 8, 3, scope='conv2', activation_fn=None)

            mask2_in = tf.concat([mask1d, dcnv2, inputs[4]], axis=3)
        else :
            mask2_in = inputs[4]

        icnv2 = slim.conv2d(mask2_in, 8, 5, scope='icnv2', activation_fn=None)
        mask2 = slim.conv2d(icnv2, num_dyn, 3, scope='mask2', activation_fn=None)
        mask2d = tf.image.resize_bilinear(mask2, [H * 8, W * 8])
        dcnv3 = slim.conv2d(mask2d, 8, 3, scope='conv3', activation_fn=None)

        mask3_in = tf.concat([mask2d, dcnv3, inputs[3]], axis=3)
        icnv3 = slim.conv2d(mask3_in, 8, 5, scope='icnv3', activation_fn=None)
        mask3 = slim.conv2d(icnv3, num_dyn, 3, scope='mask3', activation_fn=None)
        mask3d = tf.image.resize_bilinear(mask3, [H * 4, W * 4])
        dcnv4 = slim.conv2d(mask3d, 8, 3, scope='conv4', activation_fn=None)

        mask4_in = tf.concat([mask3d, dcnv4, inputs[2]], axis=3)
        icnv4 = slim.conv2d(mask4_in, 8, 3, scope='icnv4', activation_fn=None)
        mask4 = slim.conv2d(icnv4, num_dyn, 3, scope='mask4', activation_fn=None)
        mask4d = tf.image.resize_bilinear(mask4, [H * 2, W * 2])
        dcnv5 = slim.conv2d(mask4d, 8, 3, scope='conv5', activation_fn=None)

        mask5_in = tf.concat([mask4d, dcnv5, inputs[1]], axis=3)
        icnv5 = slim.conv2d(mask5_in, 8, 3, scope='icnv5', activation_fn=None)
        mask5 = slim.conv2d(icnv5, num_dyn, 3, scope='mask5', activation_fn=None)
        mask5d = tf.image.resize_bilinear(mask5, [H, W])
        dcnv6 = slim.conv2d(mask5d, 8, 3, scope='conv6', activation_fn=None)

        mask6_in = tf.concat([mask5d, dcnv6, inputs[0]], axis=3)
        icnv6 = slim.conv2d(mask6_in, 8, 3, scope='icnv6', activation_fn=None)
        mask6 = slim.conv2d(icnv6, num_dyn, 3, scope='mask6', activation_fn=None)

        pose_pred = slim.conv2d(icnv6, 6, [1, 1], scope='pred1',
            stride=1, normalizer_fn=None, activation_fn=None)
        pose_avg = 0.001 * tf.reduce_mean(pose_pred, [2, 3], keep_dims=True) # B * 6
        pose_final = tf.reshape(pose_avg, [-1, 6])

        masks = [mask2, mask3, mask4, mask5, mask6]
        if full_res:
            return [mask0 + mask1] + masks
        else :
            return masks

with tf.variable_scope('mask_dyn'):
    mask_dyn_fw, mask_dyn_bw = _mask_dyn(flows_fw[-1], flows_bw[-1])
    masks_dyn_fw.append(mask_dyn_fw)
    masks_dyn_bw.append(mask_dyn_bw)

def _mask_dyn(flows_fw, flows_bw):
    inputs_fw = []
    inputs_bw = []
    for i in range(len(flows_fw ) -1, -1, -1):
        _ , h_local, w_local, _ = tf.unstack(tf.shape(flows_fw[i]))
        disocc_fw = forward_warp(flows_fw[i])
        disocc_bw = forward_warp(flows_bw[i])
        flow_bw_warped = image_warp(flows_bw[i], flows_fw[i])
        flow_fw_warped = image_warp(flows_fw[i], flows_bw[i])
        flow_diff_fw = flows_fw[i] + flow_bw_warped
        flow_diff_bw = flows_bw[i] + flow_fw_warped

        if not train_all:
            tf.stop_gradient(disocc_fw)
            tf.stop_gradient(disocc_bw)
            tf.stop_gradient(flow_bw_warped)
            tf.stop_gradient(flow_fw_warped)
            tf.stop_gradient(flow_diff_fw)
            tf.stop_gradient(flow_diff_bw)

        input_fw = tf.concat([flows_fw[i], disocc_bw, flow_diff_fw, disocc_fw, flow_diff_bw], axis=3)
        input_fw = tf.reshape(input_fw, [num_batch, h_local, w_local, 8])
        input_bw = tf.concat([flows_bw[i], disocc_fw, flow_diff_bw, disocc_bw, flow_diff_fw], axis=3)
        input_bw = tf.reshape(input_bw, [num_batch, h_local, w_local, 8])
        inputs_fw.append(input_fw)
        inputs_bw.append(input_bw)
    mask_dyn_fw = _mask_upconv(inputs_fw)
    mask_dyn_bw = _mask_upconv(inputs_bw, reuse=True)
    return mask_dyn_fw, mask_dyn_bw

def _mask_upconv(inputs, full_res=False, reuse=False):
    with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                        reuse=reuse):
        _, H, W, _ = tf.unstack(tf.shape(inputs[0]))
        icnv6 = slim.conv2d(inputs[0], 8, 3, scope='icnv6', activation_fn=None)
        mask6 = slim.conv2d(icnv6, 1, 3, scope='mask6', activation_fn=None)
        mask6_up = tf.image.resize_bilinear(mask6, [H * 2, W * 2])
        upcnv5 = slim.conv2d_transpose(icnv6, 8, 3, stride=2, scope='deconv5')

        mask5_in = tf.concat([mask6_up, upcnv5, inputs[1]], axis=3)
        icnv5 = slim.conv2d(mask5_in, 8, 3, scope='icnv5', activation_fn=None)
        mask5 = slim.conv2d(icnv5, 1, 3, scope='mask5', activation_fn=None)
        mask5_up = tf.image.resize_bilinear(mask5, [H * 4, W * 4])
        upcnv4 = slim.conv2d_transpose(icnv5, 8, 3, stride=2, scope='deconv4')

        mask4_in = tf.concat([mask5_up, upcnv4, inputs[2]], axis=3)
        icnv4 = slim.conv2d(mask4_in, 8, 5, scope='icnv4', activation_fn=None)
        mask4 = slim.conv2d(icnv4, 1, 3, scope='mask4', activation_fn=None)
        mask4_up = tf.image.resize_bilinear(mask4, [H * 8, W * 8])
        upcnv3 = slim.conv2d_transpose(icnv4, 8, 3, stride=2, scope='deconv3')

        mask3_in = tf.concat([mask4_up, upcnv3, inputs[3]], axis=3)
        icnv3 = slim.conv2d(mask3_in, 8, 5, scope='icnv3', activation_fn=None)
        mask3 = slim.conv2d(icnv3, 1, 3, scope='mask3', activation_fn=None)
        mask3_up = tf.image.resize_bilinear(mask3, [H * 16, W * 16])
        upcnv2 = slim.conv2d_transpose(icnv3, 8, 3, stride=2, scope='deconv2')

        mask2_in = tf.concat([mask3_up, upcnv2, inputs[4]], axis=3)
        icnv2 = slim.conv2d(mask2_in, 8, 7, scope='icnv2', activation_fn=None)
        mask2 = slim.conv2d(icnv2, 1, 3, scope='mask2', activation_fn=None)

        masks = [mask2, mask3, mask4, mask5, mask6]

        if full_res:
            mask2_up = tf.image.resize_bilinear(mask2, [H * 32, W * 32])
            upcnv1 = slim.conv2d_transpose(icnv2, 8, 3, stride=2, scope='deconv1')
            mask1_in = tf.concat([mask2_up, upcnv1, inputs[5]], axis=3)
            icnv1 = slim.conv2d(mask1_in, 8, 7, scope='icnv1', activation_fn=None)
            mask1 = slim.conv2d(icnv1, 1, 3, scope='mask1', activation_fn=None)
            mask1_up = tf.image.resize_bilinear(mask1, [H * 64, W * 64])
            upcnv0 = slim.conv2d_transpose(icnv1, 8, 3, stride=2, scope='deconv0')

            mask0_in = tf.concat([mask1_up, upcnv0, inputs[6]], axis=3)
            icnv0 = slim.conv2d(mask0_in, 8, 7, scope='icnv0', activation_fn=None)
            mask0 = slim.conv2d(icnv0, 1, 3, scope='mask0', activation_fn=None)
            """
            mask0_plus = tf.clip_by_value(mask0_in, clip_value_min = 0, clip_value_max = 999)
            mask0_minus = tf.clip_by_value(mask0_in, clip_value_max = 0, clip_value_min = -999)
            mask0 = tf.sigmoid(mask1_up) * mask0_plus + (1 - tf.sigmoid(mask1_up)) * mask0_minus
            """
            masks = [mask0, mask1] + masks
        return masks
