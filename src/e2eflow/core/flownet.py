import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.layers as layers

from ..ops import correlation
from .image_warp import image_warp
import numpy as np

from .flow_util import flow_to_color
from ..ops import backward_warp, forward_warp
from .losses import occlusion, DISOCC_THRESH, create_outgoing_mask

FLOW_SCALE = 5.0
POSE_SCALING = 0.001


def flownet(im1, im2, flownet_spec='S', full_resolution=False, train_all=False,
            backward_flow=False, mask_dyn = False):
    num_batch, height, width, _ = tf.unstack(tf.shape(im1))
    flownet_num = len(flownet_spec)
    assert flownet_num > 0
    flows_fw = []
    flows_bw = []
    poses_fw = []
    poses_bw = []
    masks_dyn_fw = []
    masks_dyn_bw = []
    for i, name in enumerate(flownet_spec):
        assert name in ('C', 'c', 'S', 's', 'P', 'p')
        channel_mult = 1 if name in ('C', 'S', 'P') else 3 / 8
        full_res = full_resolution and i == flownet_num - 1

        def scoped_block():
            if name.lower() == 'c':
                assert i == 0, 'FlowNetS must be used for refinement networks'

                with tf.variable_scope('flownet_c_features'):
                    _, conv2_a, conv3_a = flownet_c_features(im1, channel_mult=channel_mult)
                    _, conv2_b, conv3_b = flownet_c_features(im2, channel_mult=channel_mult, reuse=True)

                with tf.variable_scope('flownet_c') as scope:
                    flow_fw = flownet_c(conv3_a, conv3_b, conv2_a,
                                        full_res=full_res,
                                        channel_mult=channel_mult)
                    flows_fw.append(flow_fw)
                    if backward_flow:
                        scope.reuse_variables()
                        flow_bw = flownet_c(conv3_b, conv3_a, conv2_b,
                                            full_res=full_res,
                                            channel_mult=channel_mult)
                        flows_bw.append(flow_bw)
            elif name.lower() == 's':
                def _flownet_s(im1, im2, flow=None):
                    if flow is not None:
                        flow = tf.image.resize_bilinear(flow, [height, width]) * 4 * FLOW_SCALE
                        warp = image_warp(im2, flow)
                        diff = tf.abs(warp - im1)
                        if not train_all:
                            flow = tf.stop_gradient(flow)
                            warp = tf.stop_gradient(warp)
                            diff = tf.stop_gradient(diff)

                        inputs = tf.concat([im1, im2, flow, warp, diff], axis=3)
                        inputs = tf.reshape(inputs, [num_batch, height, width, 14])
                    else:
                        inputs = tf.concat([im1, im2], 3)
                    return flownet_s(inputs,
                                     full_res=full_res,
                                     channel_mult=channel_mult)
                stacked = len(flows_fw) > 0
                with tf.variable_scope('flownet_s') as scope:
                    flow_fw = _flownet_s(im1, im2, flows_fw[-1][0] if stacked else None)
                    flows_fw.append(flow_fw)
                    if backward_flow:
                        scope.reuse_variables()
                        flow_bw = _flownet_s(im2, im1, flows_bw[-1][0]  if stacked else None)
                        flows_bw.append(flow_bw)
            elif name.lower() == 'p':
                def _flownet_p(im1, im2, flow=None):
                    if flow is not None:
                        flow = tf.image.resize_bilinear(flow, [height, width]) * 4 * FLOW_SCALE
                        warp = image_warp(im2, flow)
                        diff = tf.abs(warp - im1)
                        if not train_all:
                            flow = tf.stop_gradient(flow)
                            warp = tf.stop_gradient(warp)
                            diff = tf.stop_gradient(diff)

                        inputs = tf.concat([im1, im2, flow, warp, diff], axis=3)
                        inputs = tf.reshape(inputs, [num_batch, height, width, 14])
                    else:
                        inputs = tf.concat([im1, im2], 3)
                    return flownet_p_buf(inputs,
                                     full_res=full_res,
                                     channel_mult=channel_mult)

                stacked = len(flows_fw) > 0

                with tf.variable_scope('flownet_p') as scope:
                    flow_fw, pose_fw = _flownet_p(im1, im2, flows_fw[-1][0] if stacked else None)
                    flows_fw.append(flow_fw)
                    poses_fw.append(pose_fw)
                    if backward_flow:
                        scope.reuse_variables()
                        flow_bw, pose_bw = _flownet_p(im2, im1, flows_bw[-1][0]  if stacked else None)
                        flows_bw.append(flow_bw)
                        poses_bw.append(pose_bw)
        if i > 0:
            scope_name = "stack_{}_flownet".format(i)
            with tf.variable_scope(scope_name):
                scoped_block()
        else:
            scoped_block()

    if backward_flow & mask_dyn:
        return flows_fw, flows_bw, poses_fw, poses_bw, masks_dyn_fw, masks_dyn_bw
    elif backward_flow :
        return flows_fw, flows_bw, poses_fw, poses_bw
    return flows_fw, poses_fw


def _leaky_relu(x):
    with tf.variable_scope('leaky_relu'):
        return tf.maximum(0.1 * x, x)

def _flownet_upconv(conv6_1, conv5_1, conv4_1, conv3_1, conv2, conv1=None, inputs=None,
                    channel_mult=1, full_res=False, channels=2):
    m = channel_mult

    flow6 = slim.conv2d(conv6_1, channels, 3, scope='flow6',
                        activation_fn=None)
    deconv5 = slim.conv2d_transpose(conv6_1, int(512 * m), 4, stride=2,
                                   scope='deconv5')
    flow6_up5 = slim.conv2d_transpose(flow6, channels, 4, stride=2,
                                     scope='flow6_up5',
                                     activation_fn=None)
    concat5 = tf.concat([conv5_1, deconv5, flow6_up5], 1)
    flow5 = slim.conv2d(concat5, channels, 3, scope='flow5',
                       activation_fn=None)

    deconv4 = slim.conv2d_transpose(concat5, int(256 * m), 4, stride=2,
                                   scope='deconv4')
    flow5_up4 = slim.conv2d_transpose(flow5, channels, 4, stride=2,
                                     scope='flow5_up4',
                                     activation_fn=None)
    concat4 = tf.concat([conv4_1, deconv4, flow5_up4], 1)
    flow4 = slim.conv2d(concat4, channels, 3, scope='flow4',
                       activation_fn=None)

    deconv3 = slim.conv2d_transpose(concat4, int(128 * m), 4, stride=2,
                                   scope='deconv3')
    flow4_up3 = slim.conv2d_transpose(flow4, channels, 4, stride=2,
                                     scope='flow4_up3',
                                     activation_fn=None)
    concat3 = tf.concat([conv3_1, deconv3, flow4_up3], 1)
    flow3 = slim.conv2d(concat3, channels, 3, scope='flow3',
                       activation_fn=None)

    deconv2 = slim.conv2d_transpose(concat3, int(64 * m), 4, stride=2,
                                   scope='deconv2')
    flow3_up2 = slim.conv2d_transpose(flow3, channels, 4, stride=2,
                                     scope='flow3_up2',
                                     activation_fn=None)
    concat2 = tf.concat([conv2, deconv2, flow3_up2], 1)
    flow2 = slim.conv2d(concat2, channels, 3, scope='flow2',
                       activation_fn=None)

    flows = [flow2, flow3, flow4, flow5, flow6]

    if full_res:
        with tf.variable_scope('full_res'):
            deconv1 = slim.conv2d_transpose(concat2, int(32 * m), 4, stride=2,
                                           scope='deconv1')
            flow2_up1 = slim.conv2d_transpose(flow2, channels, 4, stride=2,
                                             scope='flow2_up1',
                                             activation_fn=None)
            concat1 = tf.concat([conv1, deconv1, flow2_up1], 1)
            flow1 = slim.conv2d(concat1, channels, 3, scope='flow1',
                                activation_fn=None)

            deconv0 = slim.conv2d_transpose(concat1, int(16 * m), 4, stride=2,
                                           scope='deconv0')
            flow1_up0 = slim.conv2d_transpose(flow1, channels, 4, stride=2,
                                             scope='flow1_up0',
                                             activation_fn=None)
            concat0 = tf.concat([inputs, deconv0, flow1_up0], 1)
            flow0 = slim.conv2d(concat0, channels, 3, scope='flow0',
                                activation_fn=None)

            flows = [flow0, flow1] + flows

    return flows


def nhwc_to_nchw(tensors):
    return [tf.transpose(t, [0, 3, 1, 2]) for t in tensors]


def nchw_to_nhwc(tensors):
    return [tf.transpose(t, [0, 2, 3, 1]) for t in tensors]


def flownet_p(inputs, channel_mult=1, full_res=False):
    """Given stacked inputs, returns flow predictions in decreasing resolution.

    Uses FlowNetSimple.
    """
    m = channel_mult
    inputs = nhwc_to_nchw([inputs])[0]

    with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                        data_format='NCHW',
                        weights_regularizer=slim.l2_regularizer(0.0004),
                        weights_initializer=layers.variance_scaling_initializer(),
                        activation_fn=_leaky_relu):
        cnv1 = slim.conv2d(inputs, 32, [7, 7], stride=2, scope='cnv1')
        cnv1b = slim.conv2d(cnv1, 32, [7, 7], stride=1, scope='cnv1b')
        cnv2 = slim.conv2d(cnv1b, 64, [5, 5], stride=2, scope='cnv2')
        cnv2b = slim.conv2d(cnv2, 64, [5, 5], stride=1, scope='cnv2b')
        cnv3 = slim.conv2d(cnv2b, 128, [3, 3], stride=2, scope='cnv3')
        cnv3b = slim.conv2d(cnv3, 128, [3, 3], stride=1, scope='cnv3b')
        cnv4 = slim.conv2d(cnv3b, 256, [3, 3], stride=2, scope='cnv4')
        cnv4b = slim.conv2d(cnv4, 256, [3, 3], stride=1, scope='cnv4b')
        cnv5 = slim.conv2d(cnv4b, 512, [3, 3], stride=2, scope='cnv5')
        cnv5b = slim.conv2d(cnv5, 512, [3, 3], stride=1, scope='cnv5b')
        cnv6 = slim.conv2d(cnv5b, 512, [3, 3], stride=2, scope='cnv6')
        cnv6b = slim.conv2d(cnv6, 512, [3, 3], stride=1, scope='cnv6b')

        icnv6 = slim.conv2d(cnv6b, 512, [3, 3], stride=1, scope='icnv6')
        pose6 = POSE_SCALING * slim.conv2d(icnv6, 6, 4, activation_fn=None, stride=1, scope='pose6')

        upcnv5 = slim.conv2d_transpose(icnv6, 512, 4, stride=2, scope='upcnv5')
        depose5 = slim.conv2d_transpose(pose6, 6, 4, stride=2, scope='depose5')
        i5_in = tf.concat([upcnv5, cnv5b, depose5], axis=1)
        icnv5 = slim.conv2d(i5_in, 512, 4, stride=1, scope='icnv5')
        pose5 = POSE_SCALING * slim.conv2d(icnv5, 6, 4, activation_fn=None, stride=1, scope='pose5')

        upcnv4 = slim.conv2d_transpose(icnv5, 256, 4, stride=2, scope='upcnv4')
        depose4 = slim.conv2d_transpose(pose5, 6, 4, stride=2, scope='depose4')
        i4_in = tf.concat([upcnv4, cnv4b, depose4], axis=1)
        icnv4 = slim.conv2d(i4_in, 256, 4, stride=1, scope='icnv4')
        pose4 = POSE_SCALING * slim.conv2d(icnv4, 6, 4, activation_fn=None, stride=1, scope='pose4')

        upcnv3 = slim.conv2d_transpose(icnv4, 128, 4, stride=2, scope='upcnv3')
        depose3 = slim.conv2d_transpose(pose4, 6, 4, stride=2, scope='depose3')
        i3_in = tf.concat([upcnv3, cnv3b, depose3], axis=1)
        icnv3 = slim.conv2d(i3_in, 128, 4, stride=1, scope='icnv3')
        pose3 = POSE_SCALING * slim.conv2d(icnv3, 6, 4, activation_fn=None, stride=1, scope='pose3')

        upcnv2 = slim.conv2d_transpose(icnv3, 64, 4, stride=2, scope='upcnv2')
        depose2 = slim.conv2d_transpose(pose3, 6, 4, stride=2, scope='depose2')
        i2_in = tf.concat([upcnv2, cnv2b, depose2], axis=1)
        icnv2 = slim.conv2d(i2_in, 64, 4, stride=1, scope='icnv2')
        pose2 = POSE_SCALING * slim.conv2d(icnv2, 6, 4, activation_fn=None, stride=1, scope='pose2')

        pose_final = [pose2, pose3, pose4, pose5, pose6]
        if full_res:
            upcnv1 = slim.conv2d_transpose(icnv2, 32, 4, stride=2, scope='upcnv1')
            depose1 = slim.conv2d_transpose(pose2, 6, 4, stride=2, scope='depose1')
            i1_in = tf.concat([upcnv1, cnv1b, depose1], axis=1)
            icnv1 = slim.conv2d(i1_in, 32, 4, stride=1, scope='icnv1')
            pose1 = POSE_SCALING * slim.conv2d(icnv1, 6, 4, activation_fn=None, stride=1, scope='pose1')

            upcnv0 = slim.conv2d_transpose(icnv1, 16, 4, stride=2, scope='upcnv0')
            depose0 = slim.conv2d_transpose(pose1, 6, 4, stride=2, scope='depose0')
            i0_in = tf.concat([upcnv0, inputs, depose0], axis=1)
            icnv0 = slim.conv2d(i0_in, 16, 4, stride=1, scope='icnv0')
            pose0 = POSE_SCALING * slim.conv2d(icnv0, 6, 4, activation_fn=None, stride=1, scope='pose0')

            inputs = tf.concat([inputs, pose0], axis=1)
            conv1 = tf.concat([cnv1b, pose1], axis=1)
            pose_final = [pose0, pose1] + pose_final
        else :
            conv1 = cnv1b
        conv2 = tf.concat([cnv2b, pose2], axis=1)
        conv3_1 = tf.concat([cnv3b, pose3], axis=1)
        conv4_1 = tf.concat([cnv4b, pose4], axis=1)
        conv5_1 = tf.concat([cnv5b, pose5], axis=1)
        conv6_1 = tf.concat([cnv6b, pose6], axis=1)

        res = _flownet_upconv(conv6_1, conv5_1, conv4_1, conv3_1, conv2, conv1, inputs,
                              channel_mult=channel_mult, full_res=full_res)
        return nchw_to_nhwc(res), nchw_to_nhwc(pose_final)

def flownet_p_buf(inputs, channel_mult=1, full_res=False):
    """Given stacked inputs, returns flow predictions in decreasing resolution.

    Uses FlowNetSimple.
    """
    m = channel_mult
    inputs = nhwc_to_nchw([inputs])[0]

    with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                        data_format='NCHW',
                        weights_regularizer=slim.l2_regularizer(0.0004),
                        weights_initializer=layers.variance_scaling_initializer(),
                        activation_fn=_leaky_relu):
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

        flows, poses = _flownet_p_upconv(conv6_1, conv5_1, conv4_1, conv3_1, conv2, conv1, inputs,
                              channel_mult=channel_mult, full_res=full_res)
        return nchw_to_nhwc(flows), nchw_to_nhwc(poses)

def _flownet_p_upconv(conv6_1, conv5_1, conv4_1, conv3_1, conv2, conv1=None, inputs=None,
                    channel_mult=1, full_res=False, channels=2, channels_p=8):
    m = channel_mult

    pose6 = slim.conv2d(conv6_1, channels_p, 3, scope='pose6',
                        activation_fn=None)
    flow6 = slim.conv2d(pose6, channels, 1, scope='flow6',
                        activation_fn=None)
    deconv5 = slim.conv2d_transpose(conv6_1, int(512 * m), 4, stride=2,
                                   scope='deconv5')
    pose6_up5 = slim.conv2d_transpose(pose6, channels_p, 4, stride=2,
                                     scope='pose6_up5',
                                     activation_fn=None)
    flow6_up5 = slim.conv2d_transpose(flow6, channels, 4, stride=2,
                                     scope='flow6_up5',
                                     activation_fn=None)
    concat5 = tf.concat([conv5_1, deconv5, pose6_up5, flow6_up5], 1)

    pose5 = slim.conv2d(concat5, channels_p, 3, scope='pose5',
                       activation_fn=None)
    flow5 = slim.conv2d(pose5, channels, 1, scope='flow5',
                        activation_fn=None)
    deconv4 = slim.conv2d_transpose(concat5, int(256 * m), 4, stride=2,
                                   scope='deconv4')
    pose5_up4 = slim.conv2d_transpose(pose5, channels_p, 4, stride=2,
                                     scope='pose5_up4',
                                     activation_fn=None)
    flow5_up4 = slim.conv2d_transpose(flow5, channels, 4, stride=2,
                                     scope='flow5_up4',
                                     activation_fn=None)
    concat4 = tf.concat([conv4_1, deconv4, pose5_up4, flow5_up4], 1)

    pose4 = slim.conv2d(concat4, channels_p, 3, scope='pose4',
                       activation_fn=None)
    flow4 = slim.conv2d(pose4, channels, 1, scope='flow4',
                        activation_fn=None)
    deconv3 = slim.conv2d_transpose(concat4, int(128 * m), 4, stride=2,
                                   scope='deconv3')
    pose4_up3 = slim.conv2d_transpose(pose4, channels_p, 4, stride=2,
                                     scope='pose4_up3',
                                     activation_fn=None)
    flow4_up3 = slim.conv2d_transpose(flow4, channels, 4, stride=2,
                                     scope='flow4_up3',
                                     activation_fn=None)
    concat3 = tf.concat([conv3_1, deconv3, pose4_up3, flow4_up3], 1)

    pose3 = slim.conv2d(concat3, channels_p, 3, scope='pose3',
                       activation_fn=None)
    flow3 = slim.conv2d(pose3, channels, 1, scope='flow3',
                        activation_fn=None)

    deconv2 = slim.conv2d_transpose(concat3, int(64 * m), 4, stride=2,
                                   scope='deconv2')
    pose3_up2 = slim.conv2d_transpose(pose3, channels_p, 4, stride=2,
                                     scope='pose3_up2',
                                     activation_fn=None)
    flow3_up2 = slim.conv2d_transpose(flow3, channels, 4, stride=2,
                                     scope='flow3_up2',
                                     activation_fn=None)
    concat2 = tf.concat([conv2, deconv2, pose3_up2, flow3_up2], 1)
    pose2 = slim.conv2d(concat2, channels_p, 3, scope='pose2',
                       activation_fn=None)
    flow2 = slim.conv2d(pose2, channels, 1, scope='flow2',
                        activation_fn=None)

    flows = [flow2, flow3, flow4, flow5, flow6]
    poses = [pose2, pose3, pose4, pose5, pose6]

    if full_res:
        with tf.variable_scope('full_res'):
            deconv1 = slim.conv2d_transpose(concat2, int(32 * m), 4, stride=2,
                                           scope='deconv1')
            pose2_up1 = slim.conv2d_transpose(pose2, channels_p, 4, stride=2,
                                             scope='pose2_up1',
                                             activation_fn=None)
            flow2_up1 = slim.conv2d_transpose(flow2, channels, 4, stride=2,
                                             scope='flow2_up1',
                                             activation_fn=None)
            concat1 = tf.concat([conv1, deconv1, pose2_up1, flow2_up1], 1)
            pose1 = slim.conv2d(concat1, channels_p, 3, scope='pose1',
                                        activation_fn=None)
            flow1 = slim.conv2d(pose1, channels, 1, scope='flow1',
                                activation_fn=None)

            deconv0 = slim.conv2d_transpose(concat1, int(16 * m), 4, stride=2,
                                           scope='deconv0')
            pose1_up0 = slim.conv2d_transpose(pose1, channels_p, 4, stride=2,
                                             scope='pose1_up0',
                                             activation_fn=None)
            flow1_up0 = slim.conv2d_transpose(flow1, channels, 4, stride=2,
                                             scope='flow1_up0',
                                             activation_fn=None)
            concat0 = tf.concat([inputs, deconv0, pose1_up0, flow1_up0], 1)
            pose0 = slim.conv2d(concat0, channels_p, 3, scope='pose0',
                                activation_fn=None)
            flow0 = slim.conv2d(pose0, channels, 1, scope='flow0',
                                activation_fn=None)

            flows = [flow0, flow1] + flows
            poses = [pose0, pose1] + poses

    return flows, poses

def flownet_s(inputs, channel_mult=1, full_res=False):
    """Given stacked inputs, returns flow predictions in decreasing resolution.

    Uses FlowNetSimple.
    """
    m = channel_mult
    inputs = nhwc_to_nchw([inputs])[0]

    with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                        data_format='NCHW',
                        weights_regularizer=slim.l2_regularizer(0.0004),
                        weights_initializer=layers.variance_scaling_initializer(),
                        activation_fn=_leaky_relu):
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

        res = _flownet_upconv(conv6_1, conv5_1, conv4_1, conv3_1, conv2, conv1, inputs,
                              channel_mult=channel_mult, full_res=full_res)
        return nchw_to_nhwc(res)


def flownet_c_features(im, channel_mult=1, reuse=None):
    m = channel_mult
    im = nhwc_to_nchw([im])[0]
    with slim.arg_scope([slim.conv2d],
                        data_format='NCHW',
                        weights_regularizer=slim.l2_regularizer(0.0004),
                        weights_initializer=layers.variance_scaling_initializer(),
                        activation_fn=_leaky_relu):
        conv1 = slim.conv2d(im, int(64 * m), 7, stride=2, scope='conv1', reuse=reuse)
        conv2 = slim.conv2d(conv1, int(128 * m), 5, stride=2, scope='conv2', reuse=reuse)
        conv3 = slim.conv2d(conv2, int(256 * m), 5, stride=2, scope='conv3', reuse=reuse)
        return conv1, conv2, conv3


def flownet_c(conv3_a, conv3_b, conv2_a, channel_mult=1, full_res=False):
    """Given two images, returns flow predictions in decreasing resolution.

    Uses FlowNetCorr.
    """
    m = channel_mult

    with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                        data_format='NCHW',
                        weights_regularizer=slim.l2_regularizer(0.0004),
                        weights_initializer=layers.variance_scaling_initializer(),
                        activation_fn=_leaky_relu):
        corr = correlation(conv3_a, conv3_b,
                           pad=20, kernel_size=1, max_displacement=20, stride_1=1, stride_2=2)

        conv_redir = slim.conv2d(conv3_a, int(32 * m), 1, stride=1, scope='conv_redir')

        conv3_1 = slim.conv2d(tf.concat([conv_redir, corr], 1), int(256 * m), 3,
                              stride=1, scope='conv3_1')
        conv4 = slim.conv2d(conv3_1, int(512 * m), 3, stride=2, scope='conv4')
        conv4_1 = slim.conv2d(conv4, int(512 * m), 3, stride=1, scope='conv4_1')
        conv5 = slim.conv2d(conv4_1, int(512 * m), 3, stride=2, scope='conv5')
        conv5_1 = slim.conv2d(conv5, int(512 * m), 3, stride=1, scope='conv5_1')
        conv6 = slim.conv2d(conv5_1, int(1024 * m), 3, stride=2, scope='conv6')
        conv6_1 = slim.conv2d(conv6, int(1024 * m), 3, stride=1, scope='conv6_1')

        res = _flownet_upconv(conv6_1, conv5_1, conv4_1, conv3_1, conv2_a,
                              channel_mult=channel_mult, full_res=full_res)
        return nchw_to_nhwc(res)
