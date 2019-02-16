from __future__ import unicode_literals, division, print_function
from config import SPACING, SUBJECTS_DIR, NN_TESTS_DIR, CHANNELS, N_DIPOLES, GROUND_TRUTH, HEMISPHERE, SUBJECT_NAME
from os.path import join
import numpy as np
import mne
from mayavi import mlab
from surfer import Brain
import math


def visualize_dle_colormaps(data, src, colormap='hot', smoothing_steps=50):

    mne_dle, sloreta_dle, dspm_dle, nn_dle = np.array([]), np.array([]), np.array([]), np.array([])
    mne_sd, sloreta_sd, dspm_sd, nn_sd = np.array([]), np.array([]), np.array([]), np.array([])
    mne_oa, sloreta_oa, dspm_oa, nn_oa = np.array([]), np.array([]), np.array([]), np.array([])

    if HEMISPHERE == 'rh':
        hemi = 1
    else:
        hemi = 0

    offset = len(CHANNELS) + len(GROUND_TRUTH) + 1  # 5
    diff = len(CHANNELS) + len(GROUND_TRUTH)

    ground_truth_verts = map(int, list(data[:, 0]))

    unique_verts = sorted(set(ground_truth_verts))

    print(unique_verts)

    for c in range(0, len(CHANNELS)):
        avgs_dle = np.zeros(max(unique_verts) + 1)
        avgs_sd = np.zeros(max(unique_verts) + 1)
        avgs_oa = np.zeros(max(unique_verts) + 1)
        counts = np.zeros(max(unique_verts) + 1)
        for d in range(0, data.shape[0]):
            vert = int(data[d, 0])
            avgs_dle[vert] += data[d, offset + c]
            avgs_sd[vert] += data[d, offset + c + diff]
            avgs_oa[vert] += data[d, offset + c + 2 * diff]
            counts[vert] += 1
        avgs_dle = avgs_dle[counts != 0] * 100
        avgs_sd = avgs_sd[counts != 0] * 100
        avgs_oa = avgs_oa[counts != 0]
        counts = counts[counts != 0]
        if CHANNELS[c] == 'mne':
            mne_dle = np.true_divide(avgs_dle, counts)
            mne_sd = np.true_divide(avgs_sd, counts)
            mne_oa = np.true_divide(avgs_oa, counts)

        elif CHANNELS[c] == 'sloreta':
            sloreta_dle = np.true_divide(avgs_dle, counts)
            sloreta_sd = np.true_divide(avgs_sd, counts)
            sloreta_oa = np.true_divide(avgs_oa, counts)

        elif CHANNELS[c] == 'dspm':
            dspm_dle = np.true_divide(avgs_dle, counts)
            dspm_sd = np.true_divide(avgs_sd, counts)
            dspm_oa = np.true_divide(avgs_oa, counts)
        else:
            raise ValueError("Channel %s not understood" % CHANNELS[c])

    for g in range(0, len(GROUND_TRUTH)):
        avgs_dle = np.zeros(max(unique_verts) + 1)
        avgs_sd = np.zeros(max(unique_verts) + 1)
        avgs_oa = np.zeros(max(unique_verts) + 1)
        counts = np.zeros(max(unique_verts) + 1)
        for d in range(0, data.shape[0]):
            vert = int(data[d, 0])
            avgs_dle[vert] += data[d, offset + g + len(CHANNELS)]
            avgs_sd[vert] += data[d, offset + g + len(CHANNELS) + diff]
            avgs_oa[vert] += data[d, offset + g + len(CHANNELS) + 2 * diff]
            counts[vert] += 1
        avgs_dle = avgs_dle[counts != 0] * 100
        avgs_sd = avgs_sd[counts != 0] * 100
        avgs_oa = avgs_oa[counts != 0]
        counts = counts[counts != 0]
        if GROUND_TRUTH[g] == 'stc':
            nn_dle = np.true_divide(avgs_dle, counts)
            nn_sd = np.true_divide(avgs_sd, counts)
            nn_oa = np.true_divide(avgs_oa, counts)

    ground_truth_verts = np.where(src[hemi]['inuse'])[0][unique_verts]
    print(ground_truth_verts)

    maxv_dle = math.ceil(max(mne_dle.max(), sloreta_dle.max(), dspm_dle.max(), nn_dle.max()))
    maxv_sd = math.ceil(max(mne_sd.max(), sloreta_sd.max(), dspm_sd.max(), nn_sd.max()))
    maxv_oa = math.ceil(max(mne_oa.max(), sloreta_oa.max(), dspm_oa.max(), nn_oa.max()))

    minv_dle = 0
    minv_sd = math.floor(min(mne_sd.min(), sloreta_sd.min(), dspm_sd.min(), nn_sd.min()))
    minv_oa = math.floor(min(mne_oa.min(), sloreta_oa.min(), dspm_oa.min(), nn_oa.min()))

    midv_dle = (maxv_dle + minv_dle) // 2
    midv_sd = (maxv_sd + minv_sd) // 2
    midv_oa = (maxv_oa + minv_oa) // 2

    print(mne_dle)
    print(mne_dle.sum() / len(mne_dle))
    print(sloreta_dle)
    print(sloreta_dle.sum() / len(sloreta_dle))
    print(dspm_dle)
    print(dspm_dle.sum() / len(dspm_dle))
    print(nn_dle)
    print(nn_dle.sum() / len(nn_dle))
    print(len(mne_dle))

    for c in CHANNELS:
        brain = Brain(SUBJECT_NAME, HEMISPHERE, 'inflated', subjects_dir=SUBJECTS_DIR)
        if c == 'mne':
            brain.add_data(mne_dle, colormap=colormap, vertices=ground_truth_verts,
                           smoothing_steps=smoothing_steps, hemi=HEMISPHERE)
            brain.scale_data_colormap(fmin=minv_dle, fmid=midv_dle, fmax=maxv_dle, transparent=True)
            mlab.savefig(join("visualization", "mne_dle_heatmap.png"))
            #mlab.show()

            brain = Brain(SUBJECT_NAME, HEMISPHERE, 'inflated', subjects_dir=SUBJECTS_DIR)
            brain.add_data(mne_sd, colormap=colormap, vertices=ground_truth_verts,
                           smoothing_steps=smoothing_steps, hemi=HEMISPHERE)
            brain.scale_data_colormap(fmin=minv_sd, fmid=midv_sd, fmax=maxv_sd, transparent=True)
            mlab.savefig(join("visualization", "mne_sd_heatmap.png"))
            #mlab.show()

            brain = Brain(SUBJECT_NAME, HEMISPHERE, 'inflated', subjects_dir=SUBJECTS_DIR)
            brain.add_data(mne_oa, colormap=colormap, vertices=ground_truth_verts,
                           smoothing_steps=smoothing_steps, hemi=HEMISPHERE)
            brain.scale_data_colormap(fmin=minv_oa, fmid=midv_oa, fmax=maxv_oa, transparent=True)
            mlab.savefig(join("visualization", "mne_oa_heatmap.png"))
            #mlab.show()

        elif c == 'sloreta':
            brain.add_data(sloreta_dle, colormap=colormap, vertices=ground_truth_verts,
                            smoothing_steps=smoothing_steps, hemi=HEMISPHERE)
            brain.scale_data_colormap(fmin=minv_dle, fmid=midv_dle, fmax=maxv_dle, transparent=True)
            mlab.savefig(join("visualization", "sloreta_dle_heatmap.png"))
            #mlab.show()

            brain = Brain(SUBJECT_NAME, HEMISPHERE, 'inflated', subjects_dir=SUBJECTS_DIR)
            brain.add_data(sloreta_sd, colormap=colormap, vertices=ground_truth_verts,
                           smoothing_steps=smoothing_steps, hemi=HEMISPHERE)
            brain.scale_data_colormap(fmin=minv_sd, fmid=midv_sd, fmax=maxv_sd, transparent=True)
            mlab.savefig(join("visualization", "sloreta_sd_heatmap.png"))
            #mlab.show()

            brain = Brain(SUBJECT_NAME, HEMISPHERE, 'inflated', subjects_dir=SUBJECTS_DIR)
            brain.add_data(sloreta_oa, colormap=colormap, vertices=ground_truth_verts,
                           smoothing_steps=smoothing_steps, hemi=HEMISPHERE)
            brain.scale_data_colormap(fmin=minv_oa, fmid=midv_oa, fmax=maxv_oa, transparent=True)
            mlab.savefig(join("visualization", "sloreta_oa_heatmap.png"))
            #mlab.show()

        elif c == 'dspm':
            brain.add_data(dspm_dle, colormap=colormap, vertices=ground_truth_verts,
                            smoothing_steps=smoothing_steps, hemi=HEMISPHERE)
            brain.scale_data_colormap(fmin=minv_dle, fmid=midv_dle, fmax=maxv_dle, transparent=True)
            mlab.savefig(join("visualization", "dspm_dle_heatmap.png"))
            #mlab.show()

            brain = Brain(SUBJECT_NAME, HEMISPHERE, 'inflated', subjects_dir=SUBJECTS_DIR)
            brain.add_data(dspm_sd, colormap=colormap, vertices=ground_truth_verts,
                           smoothing_steps=smoothing_steps, hemi=HEMISPHERE)
            brain.scale_data_colormap(fmin=minv_sd, fmid=midv_sd, fmax=maxv_sd, transparent=True)
            mlab.savefig(join("visualization", "dspm_sd_heatmap.png"))
            #mlab.show()

            brain = Brain(SUBJECT_NAME, HEMISPHERE, 'inflated', subjects_dir=SUBJECTS_DIR)
            brain.add_data(dspm_oa, colormap=colormap, vertices=ground_truth_verts,
                           smoothing_steps=smoothing_steps, hemi=HEMISPHERE)
            brain.scale_data_colormap(fmin=minv_oa, fmid=midv_oa, fmax=maxv_oa, transparent=True)
            mlab.savefig(join("visualization", "dspm_oa_heatmap.png"))
            #mlab.show()

    for g in GROUND_TRUTH:
        brain = Brain(SUBJECT_NAME, HEMISPHERE, 'inflated', subjects_dir=SUBJECTS_DIR)
        if g == 'stc':
            brain.add_data(nn_dle, colormap=colormap, vertices=ground_truth_verts,
                            smoothing_steps=smoothing_steps, hemi=HEMISPHERE)
            brain.scale_data_colormap(fmin=minv_dle, fmid=midv_dle, fmax=maxv_dle, transparent=True)
            mlab.savefig(join("visualization", "nn_dle_heatmap.png"))
            #mlab.show()

            brain = Brain(SUBJECT_NAME, HEMISPHERE, 'inflated', subjects_dir=SUBJECTS_DIR)
            brain.add_data(nn_sd, colormap=colormap, vertices=ground_truth_verts,
                           smoothing_steps=smoothing_steps, hemi=HEMISPHERE)
            brain.scale_data_colormap(fmin=minv_sd, fmid=midv_sd, fmax=maxv_sd, transparent=True)
            mlab.savefig(join("visualization", "nn_sd_heatmap.png"))
            #mlab.show()

            brain = Brain(SUBJECT_NAME, HEMISPHERE, 'inflated', subjects_dir=SUBJECTS_DIR)
            brain.add_data(nn_oa, colormap=colormap, vertices=ground_truth_verts,
                           smoothing_steps=smoothing_steps, hemi=HEMISPHERE)
            brain.scale_data_colormap(fmin=minv_oa, fmid=midv_oa, fmax=maxv_oa, transparent=True)
            mlab.savefig(join("visualization", "nn_oa_heatmap.png"))
            #mlab.show()


if N_DIPOLES != 1:
    raise ValueError("DLE visualization can only be run with N_DIPOLES=1")

if not SUBJECT_NAME:
    SUBJECT_NAME = str('sample')

data = np.genfromtxt(join(NN_TESTS_DIR, 'results_processed.csv'), delimiter=str(','))

src = mne.setup_source_space(str('sample'), spacing=SPACING, subjects_dir=SUBJECTS_DIR, add_dist=False)

visualize_dle_colormaps(data, src)

# surf = brain.geo[HEMISPHERE]

"""
vertidx = np.where(src[1]['inuse'])[0]


all_zero_dle = {2, 103, 168, 212, 243, 247, 298, 327, 355, 384, 527, 580}

almost_all_zero_dle = {22, 38, 39, 40, 53, 68, 70, 76, 89, 121, 137, 152, 190, 192, 207, 240, 244, 252,
                       292, 318, 335, 392, 492, 517, 550}

some_zero_dle = {44, 69, 78, 80, 99, 104, 105, 106, 120, 128, 148, 156, 164, 208, 223, 251, 255, 296,
                 319, 324, 331, 385, 393, 399, 400, 420, 441, 476, 490, 571, 581, 596}

easy_inds = list(all_zero_dle | almost_all_zero_dle)
print(easy_inds)
easy_inds = vertidx[easy_inds]

mlab.show()

#mlab.plot3d(surf.x[vertidx], surf.y[vertidx], surf.z[vertidx])
#mlab.points3d(surf.x[vertidx], surf.y[vertidx], surf.z[vertidx], color=(1, 1, 0), scale_factor=1.5)
#mlab.points3d(surf.x[easy_inds], surf.y[easy_inds], surf.z[easy_inds], color=(1, 0, 0), scale_factor=1.5)
#mlab.show()

"""