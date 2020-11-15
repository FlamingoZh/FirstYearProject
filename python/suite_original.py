# -*- coding: utf-8 -*-
# Copyright 2019 Brett D. Roads. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Experimental Suite.

This experimental suite contains the code necessary to reproduce the
results and figures in "Learning as the Unsupervised Alignment of
Conceptual Systems," by Brett D. Roads and Bradley C. Love.

Notes:
    * You need to modify fp_base to point to the root of the project
    repository (located at the bottom of this script).
    * If is_demo is set to True, an abbreviated experiment is run that
        produces a figure analogous to Figure 3 in the paper. The
        abbreviated experiment takes approximately 10 minutes. Setting
        `is_demo = False` (located at the bottom of this script) will
        run the experiment using the same settings as the paper. It
        should be noted that actual simulations take a couple of weeks
        to finish.

"""

import copy
import itertools
import json
import math
from pathlib import Path
import pickle
from pprint import pprint

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import rc
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
from PIL import Image
import pingouin as pg
from scipy.stats import spearmanr, ttest_ind, sem
from scipy.special import comb
from sklearn.manifold import MDS, TSNE
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.mixture import GaussianMixture

DATA_WORD = 'glove.840b'
DATA_IMAGE = 'openimage.box'
DATA_AUDIO = 'audioset'
DATA_PIXEL = 'imagenet'


def main(fp_base, is_demo):
    """Run script."""
    # Setup.
    fp_accuracy = fp_base / Path('results', 'accuracy')
    if not fp_accuracy.exists():
        fp_accuracy.mkdir(parents=True)
    fp_corr = fp_base / Path('results', 'correlation')
    if not fp_corr.exists():
        fp_corr.mkdir(parents=True)
    fp_fig = fp_base / Path('results', 'fig')
    if not fp_fig.exists():
        fp_fig.mkdir(parents=True)

    rc('text', usetex=True)

    # Run each experiment.
    # run_conceptual_example(fp_base)
    run_experiments(fp_base, is_demo)
    # run_followup_analysis(fp_base, is_demo)


def run_conceptual_example(fp_base):
    """Run conceptual example."""
    theme_blue = np.array([9, 48, 107]) / 255
    theme_red = np.array([103, 0, 12]) / 255
    color_strong = np.array([0, 0, 0])
    color_soft = np.array([.5, .5, .5])

    boundary = [.116, .381, .646]
    halfwidth = (boundary[1] - boundary[0]) / 2

    fp_results = fp_base / Path('results')

    # Define filepaths.
    fp_fig_2 = fp_results / Path('fig', 'fig_2.pdf')

    x1 = np.array([
        [.8, .51],
        [.76, .15],
        [.92, .87],
        [.03, .48],
        [.19, .31]
    ])

    x2 = np.array([
        [.27, .41],
        [.65, .07],
        [.94, .24],
        [.03, .33],
        [.87, .82]
    ])

    def simfunc(x, y):
        beta = 1
        d = np.sum((x - y)**2)**(.5)
        s = np.exp(-beta * d)
        return s

    def simmatfunc(x):
        n_item = x.shape[0]
        s = np.zeros([n_item, n_item])
        for i in range(n_item):
            for j in range(n_item):
                s[i, j] = simfunc(x[i, :], x[j, :])
        return s

    simmat1 = simmatfunc(x1)
    simmat2 = simmatfunc(x2)

    label1 = ['1', '2', '3', '4', '5']
    label2 = ['A', 'B', 'C', 'D', 'E']

    idx_a = np.array([0, 1, 2, 3, 4], dtype=int)
    idx_b = np.array([4, 1, 2, 3, 0], dtype=int)
    idx_c = np.array([2, 1, 4, 3, 0], dtype=int)
    locs_2 = np.array([1, 0, 0, 0, 1], dtype=bool)
    locs_3 = np.array([1, 0, 1, 0, 0], dtype=bool)

    simmat2_a = simmat2

    simmat2_b = simmat2[idx_b, :]
    simmat2_b = simmat2_b[:, idx_b]

    simmat2_c = simmat2[idx_c, :]
    simmat2_c = simmat2_c[:, idx_c]

    score_a = alignment_score(simmat1, simmat2_a)
    score_b = alignment_score(simmat1, simmat2_b)
    score_c = alignment_score(simmat1, simmat2_c)

    dmy_idx = np.array([0, 1, 2, 3, 4], dtype=int)
    s1_labels = np.array(['1', '2', '3', '4', '5'])
    s2_labels = np.array(['C', 'B', 'E', 'D', 'A'])
    idx3 = np.array([4, 1, 0, 3, 2], dtype=int)
    idx2 = np.array([2, 1, 0, 3, 4], dtype=int)

    fontdict_xlabel = {
        'fontsize': 10,
    }

    fontdict_xaxis = {
        'fontsize': 5,
        'color': color_soft
    }

    fontdict_yaxis = {
        'fontsize': 5,
        'color': color_soft
    }

    fontdict_corr = {
        'fontsize': 10,
        'verticalalignment': 'center',
        'horizontalalignment': 'center'
    }

    plt.rcParams['xtick.major.pad'] = '0'
    plt.rcParams['ytick.major.pad'] = '2'

    fig, ax_array = plt.subplots(2, 6, figsize=(6, 2))
    fig.text(
        boundary[0] + 0.02, .95, r'\textbf{a}', transform=fig.transFigure,
        horizontalalignment='center', verticalalignment='center'
    )
    fig.text(
        boundary[1] + 0.02, .95, r'\textbf{b}', transform=fig.transFigure,
        horizontalalignment='center', verticalalignment='center'
    )
    fig.text(
        boundary[2] + 0.02, .95, r'\textbf{c}', transform=fig.transFigure,
        horizontalalignment='center', verticalalignment='center'
    )

    line = matplotlib.lines.Line2D(
        (boundary[1], boundary[1]), (.05, .95), transform=fig.transFigure,
        color='k', linewidth=1
    )
    fig.lines.append(line)
    line = matplotlib.lines.Line2D(
        (boundary[2], boundary[2]), (.05, .95), transform=fig.transFigure,
        color='k', linewidth=1
    )
    fig.lines.append(line)

    fig.text(
        boundary[0] + halfwidth, 0.04,
        r'$\rho_{s}=$' + '{0:.2f}'.format(score_a),
        transform=fig.transFigure, fontdict=fontdict_corr
    )

    fig.text(
        boundary[1] + halfwidth, 0.04,
        r'$\rho_{s}=$' + '{0:.2f}'.format(score_b),
        transform=fig.transFigure, fontdict=fontdict_corr
    )

    fig.text(
        boundary[2] + halfwidth, 0.04,
        r'$\rho_{s}=$' + '{0:.2f}'.format(score_c),
        transform=fig.transFigure, fontdict=fontdict_corr
    )

    ax1 = ax_array[0, 0]
    ax2 = ax_array[1, 0]
    connectors(ax1, ax2, x1, x2, idx_a)
    subplot_spatial(ax1, x1, label1, theme_blue)
    ax1.set_ylabel('System 1', fontdict=fontdict_xlabel, color=theme_blue)
    subplot_spatial(ax2, x2, label2, theme_red)
    ax2.set_ylabel('System 2', fontdict=fontdict_xlabel, color=theme_red)

    ax = ax_array[0, 1]
    ax.matshow(simmat1, cmap=cm.Blues)
    ax.set_xticks(dmy_idx)
    ax.set_xticklabels(s1_labels, fontdict=fontdict_xaxis)
    ax.set_yticks(dmy_idx)
    ax.set_yticklabels(s1_labels, fontdict=fontdict_yaxis)
    ax.tick_params(axis=u'both', which=u'both', length=0)

    ax = ax_array[1, 1]
    ax.matshow(simmat2, cmap=cm.Reds)
    ax.set_xticks(dmy_idx)
    ax.set_xticklabels(s2_labels[idx3], fontdict=fontdict_xaxis)
    ax.set_yticks(dmy_idx)
    ax.set_yticklabels(s2_labels[idx3], fontdict=fontdict_yaxis)
    ax.tick_params(axis=u'both', which=u'both', length=0)

    ax1 = ax_array[0, 2]
    ax2 = ax_array[1, 2]
    connectors(ax1, ax2, x1, x2, idx_b, locs_2)
    subplot_spatial(ax1, x1, label1, theme_blue)
    subplot_spatial(ax2, x2, label2, theme_red)

    ax = ax_array[0, 3]
    ax.matshow(simmat1, cmap=cm.Blues)
    ax.set_xticks(dmy_idx)
    ax.set_xticklabels(s1_labels, fontdict=fontdict_xaxis)
    ax.set_yticks(dmy_idx)
    ax.set_yticklabels(s1_labels, fontdict=fontdict_yaxis)
    ax.tick_params(axis=u'both', which=u'both', length=0)

    ax = ax_array[1, 3]
    ax.matshow(simmat2_b, cmap=cm.Reds)
    ax.set_xticks(dmy_idx)
    ax.set_xticklabels(s2_labels[idx2], fontdict=fontdict_xaxis)
    ax.set_yticks(dmy_idx)
    ax.set_yticklabels(s2_labels[idx2], fontdict=fontdict_yaxis)
    ax.tick_params(axis=u'both', which=u'both', length=0)
    emphasize_tick(ax, 0, color_strong)
    emphasize_tick(ax, 4, color_strong)

    ax1 = ax_array[0, 4]
    ax2 = ax_array[1, 4]
    connectors(ax1, ax2, x1, x2, idx_c, locs_3)
    subplot_spatial(ax1, x1, label1, theme_blue)
    subplot_spatial(ax2, x2, label2, theme_red)

    ax = ax_array[0, 5]
    ax.matshow(simmat1, cmap=cm.Blues)
    ax.set_xticks(dmy_idx)
    ax.set_xticklabels(s1_labels, fontdict=fontdict_xaxis)
    ax.set_yticks(dmy_idx)
    ax.set_yticklabels(s1_labels, fontdict=fontdict_yaxis)
    ax.tick_params(axis=u'both', which=u'both', length=0)

    ax = ax_array[1, 5]
    ax.matshow(simmat2_c, cmap=cm.Reds)
    ax.set_xticks(dmy_idx)
    ax.set_xticklabels(s2_labels, fontdict=fontdict_xaxis)
    ax.set_yticks(dmy_idx)
    ax.set_yticklabels(s2_labels, fontdict=fontdict_yaxis)
    ax.tick_params(axis=u'both', which=u'both', length=0)
    emphasize_tick(ax, 0, color_strong)
    emphasize_tick(ax, 2, color_strong)

    fig.subplots_adjust(hspace=0.5)

    plt.savefig(
        fp_fig_2.absolute().as_posix(), format='pdf',
        bbox_inches="tight", dpi=400
    )


def subplot_spatial(ax, z, label, theme_color):
    """"""
    # Settings.
    s = 2
    fontdict = {
        'fontsize': 8,
    }

    n_item = z.shape[0]
    ax.scatter(z[:, 0], z[:, 1], s=s, facecolor='w')
    ax.set_xlim([-.05, 1.05])
    ax.set_ylim([-.05, 1.05])
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

    for idx in range(n_item):
        ax.text(
            z[idx, 0], z[idx, 1], label[idx], color=theme_color,
            horizontalalignment='center', verticalalignment='center',
            fontdict=fontdict
        )


def connectors(ax1, ax2, x1, x2, idxmap, locs=None):
    """Draw connections."""
    n_item = len(idxmap)
    if locs is None:
        locs = np.zeros(n_item, dtype=bool)

    arrow_connect_soft = dict(
        color=np.array([.8, .8, .8]),
        arrowstyle='-',
        clip_on=False,
        linestyle=(3, (2, 4))
    )

    arrow_connect_emph = dict(
        color=np.array([.4, .4, .4]),
        arrowstyle='-',
        clip_on=False,
        linestyle=(3, (2, 4))
    )

    for idx1 in range(n_item):
        idx2 = idxmap[idx1]
        if locs[idx1]:
            arrow_connect = arrow_connect_emph
        else:
            arrow_connect = arrow_connect_soft
        ax1.annotate(
            '', xy=(x1[idx1, 0], x1[idx1, 1]),
            xytext=(x2[idx2, 0], x2[idx2, 1]),
            xycoords=ax1.transData, textcoords=ax2.transData,
            arrowprops=arrow_connect
        )
        ax2.annotate(
            '', xy=(x1[idx1, 0], x1[idx1, 1]),
            xytext=(x2[idx2, 0], x2[idx2, 1]),
            xycoords=ax1.transData, textcoords=ax2.transData,
            arrowprops=arrow_connect
        )


def emphasize_tick(ax, idx, color_strong):
    """"""
    ax.get_xticklabels()[idx].set_color(color_strong)
    ax.get_yticklabels()[idx].set_color(color_strong)
    ax.get_xticklabels()[idx].set_weight('bold')
    ax.get_yticklabels()[idx].set_weight('bold')


def run_experiments(fp_base, is_demo):
    """Run Experiment 1."""
    fp_results = fp_base / Path('results')

    # Plot settings.
    color_ti = np.array([.4, 0., .8])
    color_ta = np.array([0., .56, 0.])
    color_ia = np.array([.9, .45, 0.])
    color_tp = np.array([0., 0.5, .5])

    # Analysis settings.
    # Set a variable to False if you would like to skip the analysis.
    do_correlation_word_image = True
    do_correlation_word_audio = True
    do_correlation_image_audio = True
    do_strength_word_image = True
    do_strength_word_audio = True
    do_strength_image_audio = True
    do_strength_word_image_audio = True
    do_strength_noise = True
    do_strength_word_image_aoa = True
    do_strength_word_imagenet = True

    # Figures.
    # Set a variable to False if you would like to skip remaking the figure.
    do_plot_fig_3 = True
    do_plot_fig_4 = True
    do_plot_fig_s1 = True
    do_plot_fig_s2 = True

    # Intersection data.
    fp_intersect_word_image = build_intersect_path(
        [DATA_WORD, DATA_IMAGE], fp_base, include_aoa=False
    )
    fp_intersect_word_audio = build_intersect_path(
        [DATA_WORD, DATA_AUDIO], fp_base, include_aoa=False
    )
    fp_intersect_image_audio = build_intersect_path(
        [DATA_IMAGE, DATA_AUDIO], fp_base, include_aoa=False
    )
    fp_intersect_word_image_aoa = build_intersect_path(
        [DATA_WORD, DATA_IMAGE], fp_base, include_aoa=True
    )
    fp_intersect_word_image_audio = build_intersect_path(
        [DATA_WORD, DATA_IMAGE, DATA_AUDIO], fp_base, include_aoa=False
    )

    # Correlation experiments.
    fp_correlation = fp_results / Path('correlation')
    fp_corr_word_image = fp_correlation / Path('corr_{0}-{1}.p'.format(
        DATA_WORD, DATA_IMAGE
    ))
    fp_corr_word_audio = fp_correlation / Path('corr_{0}-{1}.p'.format(
        DATA_WORD, DATA_AUDIO
    ))
    fp_corr_image_audio = fp_correlation / Path('corr_{0}-{1}.p'.format(
        DATA_IMAGE, DATA_AUDIO
    ))

    # Alignment Strength experiments.
    fp_accuracy = fp_results / Path('accuracy')
    fp_results_word_image_new = fp_accuracy / Path(
        'results_{0}-{1}.p'.format(DATA_WORD, DATA_IMAGE)
    )
    fp_results_word_audio_new = fp_accuracy / Path(
        'results_{0}-{1}.p'.format(DATA_WORD, DATA_AUDIO)
    )
    fp_results_image_audio_new = fp_accuracy / Path(
        'results_{0}-{1}.p'.format(DATA_IMAGE, DATA_AUDIO)
    )
    fp_results_image_word_audioset_p1 = fp_accuracy / Path(
        'results_{0}-{1}-{2}_p1.p'.format(DATA_WORD, DATA_IMAGE, DATA_AUDIO)
    )
    fp_results_image_word_audioset_p2 = fp_accuracy / Path(
        'results_{0}-{1}-{2}_p2.p'.format(DATA_WORD, DATA_IMAGE, DATA_AUDIO)
    )
    fp_results_word_image_aoa_p1 = fp_accuracy / Path(
        'results_aoa_{0}-{1}_p1.p'.format(DATA_WORD, DATA_IMAGE)
    )
    fp_results_word_image_aoa_p2 = fp_accuracy / Path(
        'results_aoa_{0}-{1}_p2.p'.format(DATA_WORD, DATA_IMAGE)
    )
    fp_results_noise = fp_accuracy / Path(
        'results_noise_{0}-{1}.p'.format(DATA_WORD, DATA_IMAGE)
    )

    # Figures
    fp_fig = fp_results / Path('fig')
    fp_fig_3 = fp_fig / Path('fig_3.pdf')
    fp_fig_4 = fp_fig / Path('fig_4.pdf')
    fp_fig_s1 = fp_fig / Path('fig_s1.pdf')
    fp_fig_s2 = fp_fig / Path('fig_s2.pdf')

    # Settings
    if is_demo:
        max_perm = 10
        n_perm_keep = 10
    else:
        max_perm = 10000
        n_perm_keep = 10000

    if do_correlation_word_image:
        # Load intersection.
        intersect_data = pickle.load(open(fp_intersect_word_image, 'rb'))
        z_0 = intersect_data['z_0']
        z_1 = intersect_data['z_1']
        vocab_intersect = intersect_data['vocab_intersect']

        set_size_list = np.array([10, 30, 100, 300, 434], dtype=int)
        sub_list = [
            np.array([3, 4, 5, 6, 7, 8, 9, 10], dtype=int),
            np.arange(3, 31, 1, dtype=int),
            np.arange(3, 101, 1, dtype=int),
            np.arange(3, 301, 1, dtype=int),
            np.arange(3, 435, 1, dtype=int)
        ]
        run_primary_analysis(
            [z_0, z_1], set_size_list, sub_list, fp_corr_word_image,
            max_perm=max_perm, n_perm_keep=n_perm_keep
        )

    if do_correlation_word_audio:
        # Load intersection.
        intersect_data = pickle.load(open(fp_intersect_word_audio, 'rb'))
        z_0 = intersect_data['z_0']
        z_1 = intersect_data['z_1']
        vocab_intersect = intersect_data['vocab_intersect']

        # set_size_list = np.array([10, 30, 100, 300, 332], dtype=int)
        # sub_list = [
        #     np.array([3, 4, 5, 6, 7, 8, 9, 10], dtype=int),
        #     np.arange(3, 31, 1, dtype=int),
        #     np.arange(3, 101, 1, dtype=int),
        #     np.arange(3, 301, 1, dtype=int).
        #     np.arange(3, 333, 1, dtype=int)
        # ]
        set_size_list = np.array([332], dtype=int)
        sub_list = [
            np.arange(3, 333, 1, dtype=int)
        ]
        run_primary_analysis(
            [z_0, z_1], set_size_list, sub_list, fp_corr_word_audio,
            max_perm=max_perm, n_perm_keep=n_perm_keep
        )

    if do_correlation_image_audio:
        # Load intersection.
        intersect_data = pickle.load(open(fp_intersect_image_audio, 'rb'))
        z_0 = intersect_data['z_0']
        z_1 = intersect_data['z_1']
        vocab_intersect = intersect_data['vocab_intersect']

        # set_size_list = np.array([10, 30, 100, 300, 332], dtype=int)
        # sub_list = [
        #     np.array([3, 4, 5, 6, 7, 8, 9, 10], dtype=int),
        #     np.arange(3, 31, 1, dtype=int),
        #     np.arange(3, 101, 1, dtype=int),
        #     np.arange(3, 301, 1, dtype=int).
        #     np.arange(3, 333, 1, dtype=int)
        # ]
        set_size_list = np.array([71], dtype=int)
        sub_list = [
            np.arange(3, 72, 1, dtype=int)
        ]
        run_primary_analysis(
            [z_0, z_1], set_size_list, sub_list, fp_corr_image_audio,
            max_perm=max_perm, n_perm_keep=n_perm_keep
        )

    if do_plot_fig_3:
        # Settings.
        fontdict_tick = {
            'fontsize': 8,
        }
        fontdict_label = {
            'fontsize': 8
        }
        fontdict_title = {
            'fontsize': 8
        }
        fontdict_sub = {
            'fontsize': 10
        }
        idx = -1

        fig, ax = plt.subplots(figsize=(5.5, 3.))
        fig.text(
            0.5, 0.02, 'Alignment Correlation', ha='center', va='center',
            fontdict=fontdict_label
        )

        sample_results = pickle.load(open(fp_corr_word_image, 'rb'))
        set_size_list = sample_results['set_size_list']
        sub_list = sample_results['sub_list']
        rho_list = sample_results['rho_list']
        n_mismatch_list = sample_results['n_mismatch_list']

        ax = plt.subplot2grid((2, 12), (0, 0), colspan=4)
        plot_score_vs_accuracy(
            ax, set_size_list[idx], n_mismatch_list[idx], rho_list[idx],
            color_ti
        )
        ax.set_title(
            'Text-Image\n{0} Concepts'.format(set_size_list[idx]),
            fontdict=fontdict_title
        )
        ax.set_ylabel('Mapping Accuracy', fontdict=fontdict_label)
        ax.tick_params(axis='both', labelsize=fontdict_tick['fontsize'])
        ax.text(
            -0.175, 1.25, r'\textbf{a}', horizontalalignment='center',
            verticalalignment='center', transform=ax.transAxes,
            fontdict=fontdict_sub
        )

        sample_results = pickle.load(open(fp_corr_word_audio, 'rb'))
        set_size_list = sample_results['set_size_list']
        sub_list = sample_results['sub_list']
        rho_list = sample_results['rho_list']
        n_mismatch_list = sample_results['n_mismatch_list']

        ax = plt.subplot2grid((2, 12), (0, 4), colspan=4)
        plot_score_vs_accuracy(
            ax, set_size_list[idx], n_mismatch_list[idx], rho_list[idx],
            color_ta
        )
        ax.set_title(
            'Text-Audio\n{0} Concepts'.format(set_size_list[idx]),
            fontdict=fontdict_title
        )
        ax.tick_params(axis='both', labelsize=fontdict_tick['fontsize'])
        ax.text(
            -0.175, 1.25, r'\textbf{b}', horizontalalignment='center',
            verticalalignment='center', transform=ax.transAxes,
            fontdict=fontdict_sub
        )

        sample_results = pickle.load(open(fp_corr_image_audio, 'rb'))
        set_size_list = sample_results['set_size_list']
        sub_list = sample_results['sub_list']
        rho_list = sample_results['rho_list']
        n_mismatch_list = sample_results['n_mismatch_list']

        ax = plt.subplot2grid((2, 12), (0, 8), colspan=4)
        plot_score_vs_accuracy(
            ax, set_size_list[idx], n_mismatch_list[idx], rho_list[idx],
            color_ia
        )
        ax.set_title(
            'Image-Audio\n{0} Concepts'.format(set_size_list[idx]),
            fontdict=fontdict_title
        )
        ax.tick_params(axis='both', labelsize=fontdict_tick['fontsize'])
        ax.text(
            -0.175, 1.25, r'\textbf{c}', horizontalalignment='center',
            verticalalignment='center', transform=ax.transAxes,
            fontdict=fontdict_sub
        )

        sample_results = pickle.load(open(fp_corr_word_image, 'rb'))
        set_size_list = sample_results['set_size_list']
        sub_list = sample_results['sub_list']
        rho_list = sample_results['rho_list']
        n_mismatch_list = sample_results['n_mismatch_list']

        idx = 0
        ax = plt.subplot2grid((2, 12), (1, 0), colspan=3)
        plot_score_vs_accuracy(
            ax, set_size_list[idx], n_mismatch_list[idx], rho_list[idx],
            color_ti
        )
        ax.set_title(
            'Text-Image\n{0} Concepts'.format(set_size_list[idx]),
            fontdict=fontdict_title
        )
        ax.set_ylabel('Mapping Accuracy', fontdict=fontdict_label)
        ax.tick_params(axis='both', labelsize=fontdict_tick['fontsize'])
        ax.text(
            -0.25, 1.275, r'\textbf{d}', horizontalalignment='center',
            verticalalignment='center', transform=ax.transAxes,
            fontdict=fontdict_sub
        )

        idx = 1
        ax = plt.subplot2grid((2, 12), (1, 3), colspan=3)
        plot_score_vs_accuracy(
            ax, set_size_list[idx], n_mismatch_list[idx], rho_list[idx],
            color_ti
        )
        ax.set_title(
            'Text-Image\n{0} Concepts'.format(set_size_list[idx]),
            fontdict=fontdict_title
        )
        ax.tick_params(axis='both', labelsize=fontdict_tick['fontsize'])
        ax.text(
            -0.25, 1.275, r'\textbf{e}', horizontalalignment='center',
            verticalalignment='center', transform=ax.transAxes,
            fontdict=fontdict_sub
        )

        idx = 2
        ax = plt.subplot2grid((2, 12), (1, 6), colspan=3)
        plot_score_vs_accuracy(
            ax, set_size_list[idx], n_mismatch_list[idx], rho_list[idx],
            color_ti
        )
        ax.set_title(
            'Text-Image\n{0} Concepts'.format(set_size_list[idx]),
            fontdict=fontdict_title
        )
        ax.tick_params(axis='both', labelsize=fontdict_tick['fontsize'])
        ax.text(
            -0.25, 1.275, r'\textbf{f}', horizontalalignment='center',
            verticalalignment='center', transform=ax.transAxes,
            fontdict=fontdict_sub
        )

        idx = 3
        ax = plt.subplot2grid((2, 12), (1, 9), colspan=3)
        plot_score_vs_accuracy(
            ax, set_size_list[idx], n_mismatch_list[idx], rho_list[idx],
            color_ti
        )
        ax.set_title(
            'Text-Image\n{0} Concepts'.format(set_size_list[idx]),
            fontdict=fontdict_title
        )
        ax.tick_params(axis='both', labelsize=fontdict_tick['fontsize'])
        ax.text(
            -0.25, 1.275, r'\textbf{g}', horizontalalignment='center',
            verticalalignment='center', transform=ax.transAxes,
            fontdict=fontdict_sub
        )
        
        plt.subplots_adjust(bottom=.12, right=.95, wspace=6.0, hspace=.7)
        plt.savefig(fp_fig_3, format='pdf', dpi=400)

    if not is_demo:
        # Settings.
        max_perm = 5000
        n_draw = 50

        if do_strength_word_image:
            intersect_data = pickle.load(open(fp_intersect_word_image, 'rb'))
            z_0 = intersect_data['z_0']
            z_1 = intersect_data['z_1']
            run_standard_sampler(
                [z_0, z_1], fp_results_word_image_new, max_perm=max_perm,
                n_draw=n_draw
            )

        if do_strength_word_audio:
            intersect_data = pickle.load(open(fp_intersect_word_audio, 'rb'))
            z_0 = intersect_data['z_0']
            z_1 = intersect_data['z_1']
            run_standard_sampler(
                [z_0, z_1], fp_results_word_audio_new, max_perm=max_perm,
                n_draw=n_draw
            )

        if do_strength_image_audio:
            intersect_data = pickle.load(open(fp_intersect_image_audio, 'rb'))
            z_0 = intersect_data['z_0']
            z_1 = intersect_data['z_1']
            run_standard_sampler(
                [z_0, z_1], fp_results_image_audio_new, max_perm=max_perm,
                n_draw=n_draw
            )

        if do_strength_word_image_audio:
            # Load intersection.
            intersect_data = pickle.load(
                open(fp_intersect_word_image_audio, 'rb')
            )
            z_0 = intersect_data['z_0']
            z_1 = intersect_data['z_1']
            z_2 = intersect_data['z_2']
            vocab_intersect = intersect_data['vocab_intersect']

            set_size_list = np.array(
                [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 59], dtype=int
            )

            print('Part 1: Two Systems')
            run_standard_sampler(
                [z_0, z_1], fp_results_image_word_audioset_p1,
                set_size_list=set_size_list, max_perm=max_perm, n_draw=n_draw
            )
            print('Part 2: Three Systems')
            run_standard_sampler(
                [z_0, z_1, z_2], fp_results_image_word_audioset_p2,
                set_size_list=set_size_list, max_perm=max_perm, n_draw=n_draw
            )

        if do_strength_word_imagenet:
            # Settings.
            if is_demo:
                n_group = 2
                max_perm = 10
                n_draw = 1
            else:
                n_group = 50
                max_perm = 5000
                n_draw = 1

            fp_intersect = build_intersect_path(
                [DATA_WORD, DATA_PIXEL], fp_base, include_aoa=False
            )
            intersect_data = pickle.load(open(fp_intersect, 'rb'))
            z_0 = intersect_data['z_0']
            z_1 = intersect_data['z_1']
            idx_1 = intersect_data['idx_1']
            vocab_intersect = intersect_data['vocab_intersect']
            n_concept = len(vocab_intersect)

            dmy_idx = np.arange(z_1.shape[0])
            idx_group = np.zeros([n_group, n_concept], dtype=int)
            np.random.seed(123)
            for i_concept in range(n_concept):
                locs = np.equal(idx_1, i_concept)
                idx_group[:, i_concept] = np.random.choice(
                    dmy_idx[locs], n_group, replace=False
                )

            for i_group in [2]:
                z_1g = z_1[idx_group[i_group, :], :]
                fp_results_word_imagenet = fp_accuracy / Path(
                    'results_{0}-{1}_g{2}.p'.format(
                        DATA_WORD, DATA_PIXEL, i_group
                    )
                )
                run_standard_sampler(
                    [z_0, z_1g], fp_results_word_imagenet, max_perm=max_perm,
                    n_draw=n_draw
                )

        if do_strength_word_image_aoa:
            # Settings.
            if is_demo:
                max_perm = 10
                n_draw = 1
            else:
                max_perm = 5000
                n_draw = 20
            # Load intersection.
            intersect_data = pickle.load(
                open(fp_intersect_word_image_aoa, 'rb')
            )
            z_0 = intersect_data['z_0']
            z_1 = intersect_data['z_1']
            aoa_ratings = intersect_data['aoa_ratings']

            run_standard_sampler(
                [z_0, z_1], fp_results_word_image_aoa_p2, max_perm=max_perm,
                n_draw=n_draw, aoa_ratings=aoa_ratings
            )

        if do_strength_noise:
                # Settings.
                do_check = False
                do_sim = True
                n_z_list = np.asarray([2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=int)
                n_set_size = len(n_z_list)
                n_perm = 10
                max_perm = 5000

                # Adversarial noise resulting in [.7, .75] percentile on last
                # two.
                noise_coeff_list = np.array([
                    2.10, 1.70, 1.80, 2.20, 1.75, 1.75, 1.70, 1.80, 1.85, 2.25
                ])
                seed_list = np.array([
                    1558, 1555, 1687, 1717, 1727, 1757, 1787, 1807, 1837, 1847
                ])
                n_noise = len(noise_coeff_list)

                intersect_data = pickle.load(
                    open(fp_intersect_word_image, 'rb')
                )
                z_0 = intersect_data['z_0']
                z_1 = intersect_data['z_1']

                # Fit spherical Gaussian to z_1.
                gmm = GaussianMixture(
                    n_components=1, covariance_type='spherical'
                )
                gmm.fit(z_1)
                cov = gmm.covariances_[0]
                print('Fitted covariance: {0:.6f}'.format(cov))

                # Check adversarial noise.
                if do_check:
                    print('Checking adversarial noise...')
                    for noise_idx in range(len(noise_coeff_list)):
                        z_list = []
                        z_list.append(z_0)

                        # Add noisy versions of embedding to list.
                        np.random.seed(seed_list[noise_idx])
                        z_noise = (
                            z_1 + noise_coeff_list[noise_idx] *
                            np.random.normal(
                                scale=np.sqrt(cov), size=z_1.shape
                            )
                        )
                        z_list.append(z_noise)

                        np.random.seed()
                        perm_percentile, _, _ = permutation_analysis(
                            z_list, 'last', max_perm=max_perm
                        )

                        print(
                            '  index: {0} | seed: {1} | noise: {2} | '
                            'percentile: {3:.2f}'.format(
                                noise_idx, seed_list[noise_idx],
                                noise_coeff_list[noise_idx],
                                100*perm_percentile
                            )
                        )

                if do_sim:
                    print('Performing aggregrate simulation...')
                    slba_list = np.zeros([n_set_size, n_perm])

                    # Determine permutations to test
                    np.random.seed(4323)
                    permutation_list = get_aggregate_permutations(
                        n_noise, n_perm
                    )

                    for i_set, n_z in enumerate(n_z_list):
                        for i_perm in range(n_perm):
                            z_list = []
                            z_list.append(z_0)

                            for noise_idx in range(n_z - 1):
                                # Add noisy versions of embedding to list.
                                curr_idx = permutation_list[i_perm, noise_idx]
                                np.random.seed(seed_list[curr_idx])
                                z_noise = z_1 + noise_coeff_list[curr_idx] * np.random.normal(
                                    scale=np.sqrt(cov), size=z_1.shape
                                )
                                z_list.append(z_noise)

                            np.random.seed()
                            slba = estimate_lba(z_list, max_perm=max_perm)
                            slba_list[i_set, i_perm] = slba
                            print(
                                '  {0} embeddings | draw {1} | SLB Accuracy: {2:.2f}'.format(
                                    n_z, i_perm, slba
                                )
                            )
                        print('  {0} embeddings | avg. SLB Accuracy: {1:.2f}'.format(n_z, np.mean(slba_list[i_set, :])))
                        pickle.dump(
                            {
                                'n_z_list': n_z_list,
                                'n_perm': n_perm,
                                'slba_list': slba_list,
                                'noise_coeff_list': noise_coeff_list,
                                'seed_list': seed_list
                            },
                            open(fp_results_noise, 'wb')
                        )

        if do_plot_fig_4:
            # Settings.
            marker_size = 3
            lw = 1
            fontdict_tick = {
                'fontsize': 8,
            }
            fontdict_label = {
                'fontsize': 8
            }
            fontdict_title = {
                'fontsize': 10
            }
            fontdict_sub = {
                'fontsize': 12
            }
            y_min = .7
            y_max = 1.04

            # Load three System data.
            fp_results_image_word_audioset_p1 = fp_accuracy / Path(
                'results_{0}-{1}-{2}_p1.p'.format(DATA_WORD, DATA_IMAGE, DATA_AUDIO)
            )
            fp_results_image_word_audioset_p2 = fp_accuracy / Path(
                'results_{0}-{1}-{2}_p2.p'.format(DATA_WORD, DATA_IMAGE, DATA_AUDIO)
            )

            perc_p1 = process_accuracy(fp_results_image_word_audioset_p1)
            perc_p2 = process_accuracy(fp_results_image_word_audioset_p2)
            # mixed_anova(fp_results_image_word_audioset_p1, fp_results_image_word_audioset_p2)
            # multiple_ttest(fp_results_image_word_audioset_p1, fp_results_image_word_audioset_p2)

            # Load noise data.
            noise_data = pickle.load(open(fp_results_noise, 'rb'))
            n_z_list = noise_data['n_z_list']
            acc_est = noise_data['slba_list']
            n_z_list = n_z_list[0:4]
            acc_est = acc_est[0:4, :]

            fig, ax = plt.subplots(figsize=(5.5, 2.))
            gs = GridSpec(1, 3, width_ratios=[1, 1, 1])
            # gs.update(left=0, right=1, bottom=.18, top=.825, wspace=.5)
            gs.update(left=0, right=.98, bottom=.2, top=.825, wspace=.5)

            # Cartoon.
            ax1 = plt.subplot(gs[0, 0])
            img = Image.open(fp_base / Path("results/fig/fig_viewpoint.jpg"))
            ax1.imshow(img)
            ax1.text(
                0, 1.2, r'\textbf{a}', horizontalalignment='center',
                verticalalignment='center', transform=ax1.transAxes,
                fontdict=fontdict_sub
            )
            ax1.axis('off')

            # Three Systems.
            ax2 = plt.subplot(gs[0, 1])
            ax2.errorbar(
                perc_p1['set_size_list'][1:], perc_p2['accuracy']['mean'][1:] - perc_p1['accuracy']['mean'][1:],
                np.sqrt(perc_p1['accuracy']['sem'][1:]**2 + perc_p2['accuracy']['sem'][1:]**2), ecolor='r',
                color=color_ti,
                marker='o', markersize=marker_size,
                linestyle='--', linewidth=lw,
                label='Text-Image'
            )
            ax2.set_xlabel('Number of Concepts', fontdict=fontdict_label)
            ax2.set_title('Two versus Three\nSystems', fontdict=fontdict_title)
            ax2.set_ylabel('Alignment Strength\nDifference', fontdict=fontdict_label)
            ax2.set_xticks(perc_p1['set_size_list'][1::2])
            ax2.set_xticklabels(perc_p1['set_size_list'][1::2])
            ax2.tick_params(axis='both', labelsize=fontdict_tick['fontsize'])
            ax2.text(
                -0.4, 1.2, r'\textbf{b}', horizontalalignment='center',
                verticalalignment='center', transform=ax2.transAxes,
                fontdict=fontdict_sub
            )

            # Noise analysis.
            ax3 = plt.subplot(gs[0, 2])
            ax3.plot(
                n_z_list, np.mean(acc_est, axis=1),
                color=color_ti,
                marker='o', markersize=marker_size,
                linestyle='--', linewidth=lw,
            )
            ax3.set_xlabel('Number of Embeddings', fontdict=fontdict_label)
            ax3.set_ylabel('Alignment Strength', fontdict=fontdict_label)
            ax3.set_title('Text-Image\nNoisy Embeddings', fontdict=fontdict_title)
            ax3.set_ylim(.8, 1.04)
            ax3.tick_params(axis='both', labelsize=fontdict_tick['fontsize'])
            ax3.text(
                -0.3, 1.2, r'\textbf{c}', horizontalalignment='center',
                verticalalignment='center', transform=ax3.transAxes,
                fontdict=fontdict_sub
            )

            # Format and save.
            plt.savefig(fp_fig_4, format='pdf', dpi=400)

        if do_plot_fig_s1:
            # Settings.
            marker_size = 3
            lw = 1
            fontdict_tick = {
                'fontsize': 8,
            }
            fontdict_label = {
                'fontsize': 8
            }
            fontdict_title = {
                'fontsize': 10
            }
            fontdict_sub = {
                'fontsize': 12
            }
            y_min = .3
            y_max = 1.05

            fig, ax = plt.subplots(figsize=(3., 2.))

            perc_word_image = process_accuracy(fp_results_word_image_new)
            perc_word_audio = process_accuracy(fp_results_word_audio_new)
            perc_image_audio = process_accuracy(fp_results_image_audio_new)

            # Load word-imagenet data.
            fp_list = []
            for i_group in np.arange(25):
                fp_list.append(
                    fp_accuracy / Path('results_{0}-{1}_g{2}.p'.format(DATA_WORD, DATA_PIXEL, i_group))
                )
            r_imagenet = collate_accuracy(fp_list)

            # Word-Image subplot.
            ax = plt.subplot(1, 1, 1)
            plt.errorbar(
                perc_word_image['set_size_list'], perc_word_image['accuracy']['mean'],
                perc_word_image['accuracy']['sem'], ecolor='r',
                color=color_ti,
                marker='o', markersize=marker_size,
                linestyle='--', linewidth=lw,
                label='Text-Image'
            )
            plt.errorbar(
                perc_word_audio['set_size_list'], perc_word_audio['accuracy']['mean'],
                perc_word_audio['accuracy']['sem'], ecolor='r',
                color=color_ta,
                marker='o', markersize=marker_size,
                linestyle='--', linewidth=lw,
                label='Text-Audio'
            )
            plt.errorbar(
                perc_image_audio['set_size_list'], perc_image_audio['accuracy']['mean'],
                perc_image_audio['accuracy']['sem'], ecolor='r',
                color=color_ia,
                marker='o', markersize=marker_size,
                linestyle='--', linewidth=lw,
                label='Image-Audio'
            )
            plt.errorbar(
                r_imagenet['set_size_list'], r_imagenet['accuracy']['mean'],
                r_imagenet['accuracy']['sem'], ecolor='r',
                color=color_tp,
                marker='o', markersize=marker_size,
                linestyle='--', linewidth=lw,
                label='Text-Pixel'
            )
            ax.set_xlabel('Number of Concepts', fontdict=fontdict_label)
            ax.set_ylabel('Alignment Strength', fontdict=fontdict_label)
            ax.set_title('Alignment of Two Systems', fontdict=fontdict_title)
            ax.set_ylim(y_min, y_max)
            ax.set_xticks([perc_word_image['set_size_list'][0], r_imagenet['set_size_list'][-1]])
            ax.set_xticklabels([perc_word_image['set_size_list'][0], r_imagenet['set_size_list'][-1]])
            ax.tick_params(axis='both', labelsize=fontdict_tick['fontsize'])
            ax.legend(fontsize=6)

            # Format and save.
            plt.tight_layout()
            plt.savefig(fp_fig_s1, format='pdf', dpi=400)

        if do_plot_fig_s2:
            # Settings.
            marker_size = 3
            lw = 1
            fontdict_tick = {
                'fontsize': 8
            }
            fontdict_label = {
                'fontsize': 8
            }
            fontdict_title = {
                'fontsize': 10
            }
            fontdict_sub = {
                'fontsize': 12
            }

            # Load AoA data.
            aoa_p1 = process_accuracy(fp_results_word_image_aoa_p1)
            aoa_p2 = process_accuracy(fp_results_word_image_aoa_p2)
            multiple_ttest(
                fp_results_word_image_aoa_p1, fp_results_word_image_aoa_p2
            )
            # Holm-Bonferroni
            # Rank p-values: 0.000, 0.000, 0.004, 0.006, .938
            # accept/reject level: .05 / (5 + 1 - np.arange(5) + 1) = array([0.00714286, 0.00833333, 0.01, 0.0125, 0.01666667])

            fig, ax = plt.subplots(figsize=(3, 2.))

            # Age-of-Acquisition.
            ax = plt.subplot(1, 1, 1)
            plt.errorbar(
                aoa_p1['set_size_list'], aoa_p2['accuracy']['mean'] - aoa_p1['accuracy']['mean'],
                np.sqrt(aoa_p1['accuracy']['sem']**2 + aoa_p2['accuracy']['sem']**2), ecolor='r',
                color=color_ti,
                marker='o', markersize=marker_size,
                linestyle='--', linewidth=lw,
                label='Unconstrained'
            )

            ax.set_xlabel('Number of Concepts', fontdict=fontdict_label)
            ax.set_ylabel(
                'Alignment Strength \nDifference', fontdict=fontdict_label
            )
            ax.set_title(
                'Text-Image\nAge-of-Acquisition', fontdict=fontdict_title
            )
            ax.set_xticks(
                [aoa_p1['set_size_list'][0], aoa_p1['set_size_list'][-1]]
            )
            ax.set_xticklabels(
                [aoa_p1['set_size_list'][0], aoa_p1['set_size_list'][-1]]
            )
            ax.tick_params(axis='both', labelsize=fontdict_tick['fontsize'])

            # Format and save.
            plt.tight_layout()
            plt.savefig(fp_fig_s2, format='pdf', dpi=400)


def run_followup_analysis(fp_base, is_demo):
    """Run new experiment."""
    fp_results = fp_base / Path('results')

    fp_intersect_word_image = build_intersect_path(
        [DATA_WORD, DATA_IMAGE], fp_base, include_aoa=False
    )

    intersect_data = pickle.load(open(fp_intersect_word_image, 'rb'))
    z_0 = intersect_data['z_0']
    z_1 = intersect_data['z_1']
    vocab_intersect = intersect_data['vocab_intersect']

    df = pd.read_csv(fp_base / Path('intersect/intersect_glove.840b-openimage.box_labels.txt'), sep=' ')
    loc_bird = df['bird'].values.astype('bool')
    loc_music = df['instrument'].values.astype('bool')
    loc_cat = df['cat'].values.astype('bool')
    loc_fruit = df['fruit'].values.astype('bool')
    n_bird = np.sum(loc_bird)
    n_fruit = np.sum(loc_fruit)
    n_music = np.sum(loc_music)

    print('Bird vs Music')
    pc_bird, pc_music, pc_bird_music = examine_swap(z_0, z_1, loc_bird, loc_music)
    print('Bird vs Fruit')
    _, pc_fruit, pc_bird_fruit = examine_swap(z_0, z_1, loc_bird, loc_fruit)
    print('Fruit vs Music')  
    _, _, pc_fruit_music = examine_swap(z_0, z_1, loc_fruit, loc_music)

    pc_within = np.hstack((pc_bird, pc_music, pc_fruit))
    pc_within_avg = np.mean(pc_within)
    pc_across = np.hstack((pc_bird_music, pc_bird_fruit, pc_fruit_music))
    pc_across_avg = np.mean(pc_across)
    print('Within PC Avg.: {0:.2f}'.format(pc_within_avg))
    print('Across PC Avg.:{0:.2f}'.format(pc_across_avg))
    print('Relative PC change: {0:.2f}'.format(pc_across_avg / pc_within_avg))


def examine_swap(z_0, z_1, loc_a, loc_b):
    """Examine proposed within versus between category swaps."""
    n_item = z_0.shape[0]
    dmy_idx = np.arange(n_item)

    idx_a = dmy_idx[loc_a]
    n_a = len(idx_a)
    idx_b = dmy_idx[loc_b]
    n_b = len(idx_b)
    n_within_max = np.maximum(n_a, n_b)

    simmat_0 = cosine_similarity(z_0)
    simmat_1 = cosine_similarity(z_1)

    # Initial.
    rho_perfect = alignment_score(simmat_0, simmat_1)

    # All possible within category swaps.
    rho_a_list = within_swap(simmat_0, simmat_1, idx_a)
    rho_a = np.mean(rho_a_list)

    rho_b_list = within_swap(simmat_0, simmat_1, idx_b)
    rho_b = np.mean(rho_b_list)


    c = list(itertools.permutations(np.arange(n_within_max, dtype=int), 2))
    c = np.array(c, dtype=int)
    if n_a > n_b:
        bad_locs = np.greater(c[:, 1], n_b - 1)
        good_locs = np.logical_not(bad_locs)
        c = c[good_locs, :]
    else:
        bad_locs = np.greater(c[:, 0], n_a - 1)
        good_locs = np.logical_not(bad_locs)
        c = c[good_locs, :]
    c = np.unique(c, axis=1)

    list_swap_idx = np.hstack(
        (
            np.expand_dims(idx_a[c[:, 0]], axis=1),
            np.expand_dims(idx_b[c[:, 1]], axis=1)
        )
    )
    n_swap = list_swap_idx.shape[0]

    rho_ab_list = np.ones(n_swap)
    for i_swap in range(n_swap):
        perm_idx = np.arange(n_item)
        perm_idx[list_swap_idx[i_swap, 0]] = list_swap_idx[i_swap, 1]
        perm_idx[list_swap_idx[i_swap, 1]] = list_swap_idx[i_swap, 0]
        simmat_1b = symmetric_matrix_indexing(simmat_1, perm_idx)
        rho_ab_list[i_swap] = alignment_score(simmat_0, simmat_1b)
    rho_ab = np.mean(rho_ab_list)

    print('perfect: {0:.6f} | A: {1:.6f} | B: {2:.6f} | AB: {3:.6f}'.format(rho_perfect, rho_a, rho_b, rho_ab))
    print(
        'A: {0:.2g}% | B: {1:.2g}% | AB: {2:.2g}%'.format(
            (rho_a - rho_perfect) / rho_perfect * 100,
            (rho_b - rho_perfect) / rho_perfect * 100,
            (rho_ab - rho_perfect) / rho_perfect * 100
        )
    )

    pc_a = (rho_a_list - rho_perfect) / rho_perfect * 100
    pc_b = (rho_b_list - rho_perfect) / rho_perfect * 100
    pc_ab = (rho_ab_list - rho_perfect) / rho_perfect * 100
    return pc_a, pc_b, pc_ab


def within_swap(simmat_0, simmat_1, idx_a):
    """Determine on the within group swaps."""
    n_item = simmat_0.shape[0]

    list_swap_idx = list(itertools.combinations(idx_a, 2))
    list_swap_idx = np.array(list_swap_idx, dtype=int)
    n_swap = list_swap_idx.shape[0]

    rho_list = np.ones(n_swap)
    for i_swap in range(n_swap):
        perm_idx = np.arange(n_item)
        perm_idx[list_swap_idx[i_swap, 0]] = list_swap_idx[i_swap, 1]
        perm_idx[list_swap_idx[i_swap, 1]] = list_swap_idx[i_swap, 0]
        simmat_1b = symmetric_matrix_indexing(simmat_1, perm_idx)
        rho_list[i_swap] = alignment_score(simmat_0, simmat_1b)
    
    return rho_list


def matrix_min(d):
    """Find minimum value in two-dimensional matrix."""
    val = np.inf
    idx_i = np.inf
    idx_j = np.inf

    n_item = d.shape[0]
    for curr_idx_i in range(n_item):
        curr_row = d[curr_idx_i, :]
        curr_idx_j = np.argmin(curr_row)
        curr_val = curr_row[curr_idx_j]
        if curr_val < val:
            idx_i = curr_idx_i
            idx_j = curr_idx_j
            val = curr_val

    return (val, idx_i, idx_j)


def estimate_lba(z_list, max_perm=1000):
    """Estimate lower-bound accuracy of structural alignment.

    Note that the first embedding defines the number of concepts.
    """
    # Settings.
    max_patience = 3
    patience_count = 0
    n_concept = z_list[0].shape[0]
    lowest_mismatch = 0

    # Evaluate mismatch of 2.
    n_mismatch = 2
    _, rho_array_2, _ = permutation_analysis(
        z_list, 'last', max_perm=max_perm
    )
    rho_correct = rho_array_2[0]
    rho_incorrect_x = rho_array_2[1:]
    # Check.
    if np.sum(np.greater(rho_incorrect_x, rho_correct)) > 0:
        # Some incorrect mappings better than correct mapping.
        patience_count = 0
        lowest_mismatch = n_mismatch
    else:
        # Correct mapping better than all incorrect mappings.
        patience_count = patience_count + 1
    n_mismatch = n_mismatch + 1

    # Search for lower bound accuracy (i.e., highest mismatch).
    better_than_correct = True
    while better_than_correct:
        # Evaluate mismatch of n_mismatch
        _, rho_array_x, _ = permutation_analysis(
            z_list, 'sub', max_perm=max_perm, n_unknown=n_mismatch,
            is_exact=True
        )
        rho_incorrect_x = rho_array_x[1:]

        # Check.
        if np.sum(np.greater(rho_incorrect_x, rho_correct)) > 0:
            # Some incorrect mappings better than correct mapping.
            patience_count = 0
            lowest_mismatch = n_mismatch
        else:
            # Correct mapping better than all incorrect mappings.
            patience_count = patience_count + 1
            if patience_count >= max_patience:
                break
        n_mismatch = n_mismatch + 1
        if n_mismatch > n_concept:
            break

    return (n_concept - lowest_mismatch) / n_concept


def estimate_accuracy(z_list, max_perm=1000):
    """Estimate accuracy of structural alignment.

    Note that the first embedding defines the number of concepts.
    """
    # Settings.
    max_patience = 3
    patience_count = 0
    n_concept = z_list[0].shape[0]

    n_mismatch_array = np.arange(0, n_concept + 1)
    n_better_array = np.zeros(n_concept + 1)

    # Evaluate mismatch of 2.
    n_mismatch = 2
    _, rho_array_2, _ = permutation_analysis(
        z_list, 'last', max_perm=max_perm
    )
    rho_correct = rho_array_2[0]
    rho_incorrect_x = rho_array_2[1:]
    locs_better = np.greater(rho_incorrect_x, rho_correct)
    n_better = np.sum(locs_better)
    # Check.
    if n_better > 0:
        # Some incorrect mappings better than correct mapping.
        patience_count = 0
        n_better_array[n_mismatch] = n_better
    else:
        # Correct mapping better than all incorrect mappings.
        patience_count = patience_count + 1

    # Search for lower bound accuracy (i.e., highest mismatch).
    n_mismatch = n_mismatch + 1
    better_than_correct = True
    while better_than_correct:
        # Evaluate mismatch of n_mismatch
        _, rho_array_x, _ = permutation_analysis(
            z_list, 'sub', max_perm=max_perm, n_unknown=n_mismatch,
            is_exact=True
        )
        rho_incorrect_x = rho_array_x[1:]
        locs_better = np.greater(rho_incorrect_x, rho_correct)
        n_better = np.sum(locs_better)

        # Check.
        if n_better > 0:
            # Some incorrect mappings better than correct mapping.
            patience_count = 0
            n_better_array[n_mismatch] = n_better
        else:
            # Correct mapping better than all incorrect mappings.
            patience_count = patience_count + 1
            if patience_count >= max_patience:
                break
        n_mismatch = n_mismatch + 1
        if n_mismatch > n_concept:
            break

    accuracy_array = (n_concept - n_mismatch_array) / n_concept
    if np.sum(n_better_array) == 0:
        n_better_array[0] = 1
    avg_accuracy = np.sum(accuracy_array * n_better_array) / np.sum(n_better_array)

    return avg_accuracy, n_better_array


def run_primary_analysis(
        z_list, set_size_list, sub_list, fp_results, max_perm=2000,
        n_perm_keep=1000):
    """"""
    z_0 = z_list[0]
    z_1 = z_list[1]
    n_item = z_0.shape[0]

    sample_results = {
        'set_size_list': set_size_list, 'sub_list': sub_list, 'rho_list': [], 'n_mismatch_list': []
    }
    print('Conditional Accuracy Analysis')
    for set_idx, set_size in enumerate(set_size_list):
        print('  {0} | set size = {1}'.format(set_idx, set_size))

        keep_idx = np.random.permutation(n_item)
        keep_idx = keep_idx[0:set_size]
        z_0_sub = z_0[keep_idx, :]
        z_1_sub = z_1[keep_idx, :]

        _, rho_array_2, perm_list_2 = permutation_analysis(
            [z_0_sub, z_1_sub], 'last', max_perm=n_perm_keep
        )
        rho_array_2 = rho_array_2[0:n_perm_keep + 1]
        n_mismatch_2 = infer_mistmatch(perm_list_2)[0][0:n_perm_keep + 1]

        rho_array = rho_array_2
        n_mismatch = n_mismatch_2

        for val_sub in sub_list[set_idx]:

            _, rho_array_sub, perm_list_sub = permutation_analysis(
                [z_0_sub, z_1_sub], 'sub', max_perm=max_perm, n_unknown=val_sub,
                is_exact=True
            )
            n_mismatch_sub = infer_mistmatch(perm_list_sub)[0]
            # Filter down to unique permutations.
            (_, keep_idx) = np.unique(perm_list_sub[0], return_index=True, axis=0)
            rho_array_uniq = rho_array_sub[keep_idx]
            n_mismatch_uniq = n_mismatch_sub[keep_idx]
            print('  mismatch {0} | unique count: {1}'.format(val_sub, len(rho_array_uniq)))
            rho_array = np.hstack((rho_array, rho_array_uniq[1:]))
            n_mismatch = np.hstack((n_mismatch, n_mismatch_uniq[1:]))

        sample_results['rho_list'].append(rho_array)
        sample_results['n_mismatch_list'].append(n_mismatch)
    print('Saving analysis results...')
    pickle.dump(sample_results, open(fp_results, 'wb'))


def run_standard_sampler(z_list, fp_results, max_perm=1000, n_draw=100, set_size_list=None, aoa_ratings=None):
    """"""
    # Settings
    draw_buffer = 5
    z_0 = z_list[0]

    # Set up sample parameters.
    handle_set_size = False
    if set_size_list is None:
        handle_set_size = True
        set_size_list = np.array([10, 30, 100, 300, 1000, 3000], dtype=int)
    aoa_set_size_list = set_size_list + draw_buffer

    # Make sure user didn't request set sizes that exceed available concepts.
    n_concept = z_0.shape[0]
    locs = np.less(set_size_list, n_concept - draw_buffer)
    set_size_list = set_size_list[locs]
    aoa_set_size_list = aoa_set_size_list[locs]

    # Append maximum if set size was not specified by user.
    if handle_set_size:
        set_size_list = np.append(set_size_list, [n_concept - draw_buffer])
        aoa_set_size_list = np.append(aoa_set_size_list, [n_concept])
    n_draw_list = n_draw * np.ones(len(set_size_list), dtype=int)

    prop_results = {
        'set_size_list': set_size_list, 'n_draw_list': n_draw_list,
        'accuracy': [], 'n_better_list': []
    }
    print('Permutation Analysis')
    for set_idx, set_size in enumerate(set_size_list):
        print('  {0} | set size = {1}'.format(set_idx, set_size))

        if aoa_ratings is not None:
            eligable_idx = determine_aoa_set(
                aoa_ratings, aoa_set_size_list[set_idx]
            )
        else:
            eligable_idx = np.arange(n_concept)

        acc_list = []
        n_better_list = []
        for _ in np.arange(n_draw_list[set_idx]):

            keep_idx = np.random.permutation(eligable_idx)
            keep_idx = keep_idx[0:set_size]
            z_list_sub = []
            for z_i in z_list:
                z_list_sub.append(z_i[keep_idx, :])

            acc, n_better_array = estimate_accuracy(z_list_sub, max_perm=max_perm)
            print('      {0:.2f}'.format(acc))
            acc_list.append(acc)
            n_better_list.append(n_better_array)

        prop_results['accuracy'].append(acc_list)
        prop_results['n_better_list'].append(n_better_list)
        print('    Mean Accuracy: {0:.2f}'.format(np.mean(acc_list)))
    print('Saving analysis results...')
    pickle.dump(prop_results, open(fp_results, 'wb'))


def plot_score_vs_accuracy(ax, set_size, n_mismatch_array, rho_array, c):
    """"""
    accuracy_array = (set_size - n_mismatch_array) / set_size

    mismatch_list = np.unique(n_mismatch_array)
    mismatch_list = mismatch_list[1:]
    n_val = len(mismatch_list)

    loc = np.equal(n_mismatch_array, 0)
    rho_correct = rho_array[loc]
    rho_correct = rho_correct[0]

    score_mean = np.zeros(n_val)
    score_std = np.zeros(n_val)
    score_min = np.zeros(n_val)
    score_max = np.zeros(n_val)
    for idx_mismatch, val_mismatch in enumerate(mismatch_list):
        loc = np.equal(n_mismatch_array, val_mismatch)
        score_mean[idx_mismatch] = np.mean(rho_array[loc])
        score_std[idx_mismatch] = np.std(rho_array[loc])
        score_min[idx_mismatch] = np.min(rho_array[loc])
        score_max[idx_mismatch] = np.max(rho_array[loc])

    accuracy = (set_size - mismatch_list) / set_size

    rho, p_val = spearmanr(accuracy_array, rho_array)
    print('rho: {0:.2f} (p={1:.4f})'.format(rho, p_val))

    ax.plot(
        score_mean, accuracy, color=c,
        # marker='o', markersize=.5,
        linestyle='-', linewidth=.5,
    )
    ax.fill_betweenx(
        accuracy, score_mean - score_std, score_mean + score_std,
        facecolor=c, alpha=.2, edgecolor='none'
    )
    ax.fill_betweenx(
        accuracy, score_min, score_max,
        facecolor=c, alpha=.2, edgecolor='none'
    )

    factor = 20
    score_beat_correct = linear_interpolation(score_max, factor=factor)
    accuracy_interp = linear_interpolation(accuracy, factor=factor)
    score_correct = rho_correct * np.ones(len(score_beat_correct))
    locs = np.less(score_beat_correct, score_correct)
    score_beat_correct[locs] = rho_correct
    ax.fill_betweenx(
        accuracy_interp, score_correct, score_beat_correct,
        facecolor='r', alpha=.75, edgecolor='none'
    )

    ax.scatter(
        rho_correct, 1.0,
        s=6, marker='x',
        color=c
    )

    ax.set_yticks([0., .5, 1.])
    ax.set_yticklabels([0., .5, 1.])


def linear_interpolation(y, factor=10):
    """Interpolate additional points for piece-wise linear function."""
    n_point = len(y)
    y_interp = np.array([])
    for idx in range(1, n_point):
        # start_x = idx - 1
        # end_x = idx
        start_y = y[idx - 1]
        end_y = y[idx]
        y_interp = np.hstack((
            y_interp,
            np.linspace(start_y, end_y, factor, endpoint=False)
        ))
    y_interp = np.asarray(y_interp)
    return y_interp


def multiple_ttest(fp_results_a, fp_results_b):
    """Perform multiple t-tests."""
    results_a = pickle.load(open(fp_results_a, 'rb'))
    results_b = pickle.load(open(fp_results_b, 'rb'))

    set_size_list = results_a['set_size_list']
    n_set_size = len(set_size_list)
    for idx in range(n_set_size):
        print('{0}'.format(set_size_list[idx]))

        mu_b = np.mean(results_b['accuracy'][idx])
        std_b = np.std(results_b['accuracy'][idx])
        print('(M={0:.3f}, SD={1:.3f})'.format(mu_b, std_b))

        mu_a = np.mean(results_a['accuracy'][idx])
        std_a = np.std(results_a['accuracy'][idx])
        print('(M={0:.3f}, SD={1:.3f})'.format(mu_a, std_a))
        print_ttest(results_b['accuracy'][idx], results_a['accuracy'][idx])


def mixed_anova(fp_results_a, fp_results_b):
    """Perform a mixed ANOVA."""
    results_a = pickle.load(open(fp_results_a, 'rb'))
    results_b = pickle.load(open(fp_results_b, 'rb'))

    set_size_list = results_a['set_size_list']
    n_set_size = len(set_size_list)

    n = 50
    months = ['5', '10', '15', '20', '25', '30', '35', '40', '45', '50', '55']
    n_set = len(months)

    control_s = np.asarray(results_a['accuracy'])
    meditation_s = np.asarray(results_b['accuracy'])

    control = np.ravel(control_s)
    meditation = np.ravel(meditation_s)
    scores = np.r_[control, meditation]

    set_size = np.r_[np.repeat(months, n), np.repeat(months, n)]
    group = np.repeat(['2-System', '3-System'], n_set * n)
    subject_id = np.r_[np.tile(np.arange(n), n_set), np.tile(np.arange(n, n + n), n_set)]

    # Create a dataframe.
    df = pd.DataFrame({
        'Scores': scores,
        'Time': set_size,
        'Group': group,
        'Subject': subject_id
    })

    # Compute the two-way mixed-design ANOVA
    aov = pg.mixed_anova(dv='Scores', within='Time', between='Group', subject='Subject', data=df)
    # Pretty printing of ANOVA summary
    pg.print_table(aov)

    posthocs = pg.pairwise_ttests(
        dv='Scores', within='Time', between='Group',
        data=df, padjust='holm'
    )
    pg.print_table(posthocs)


def collate_accuracy(fp_list):
    """"""
    n_group = len(fp_list)
    results = pickle.load(open(fp_list[0], 'rb'))
    set_size_list = results['set_size_list']
    n_set_size = len(set_size_list)

    m_all = np.zeros([n_group, n_set_size])
    for i_group, fp in enumerate(fp_list):
        results = pickle.load(open(fp, 'rb'))
        for i_set_size in range(n_set_size):
            m_all[i_group, i_set_size] = results['accuracy'][i_set_size][0]

    m_mean = np.mean(m_all, axis=0)
    m_sem = sem(m_all, axis=0)
    r = {
        'set_size_list': set_size_list,
        'accuracy': {'mean': m_mean, 'sem': m_sem}
    }
    return r


def process_accuracy(fp_results):
    """"""
    results = pickle.load(open(fp_results, 'rb'))
    set_size_list = results['set_size_list']
    n_set_size = len(set_size_list)

    res_mean = np.zeros([n_set_size])
    res_sem = np.zeros([n_set_size])
    for idx in range(n_set_size):
        res_mean[idx] = np.mean(results['accuracy'][idx])
        res_sem[idx] = sem(results['accuracy'][idx])

    r = {
        'set_size_list': set_size_list,
        'accuracy': {'mean': res_mean, 'sem': res_sem}
    }
    return r


def get_aggregate_permutations(n_noise, n_perm):
    """"""
    perms = np.array(list(itertools.permutations(np.arange(n_noise, dtype=int), n_noise)), dtype=int)
    rnd_idx = np.random.permutation(len(perms))
    perms = perms[rnd_idx]
    perms = perms[0:n_perm]
    return perms


def determine_aoa_set(aoa_ratings, n_aoa, sort='best'):
    """"""
    if sort == 'best':
        sort_idx = np.argsort(aoa_ratings)
    else:
        sort_idx = np.argsort(-aoa_ratings)
    eligable_idx = sort_idx[0:n_aoa]
    return eligable_idx


def permutation_analysis(z_list, perm_type, max_perm=10000, n_unknown=2, n_known=0, is_exact=False):
    """Perform permutation analysis.

    This function assumes embeddings are passed in correctly aligned.
    """
    # Pre-compute similarity matrices.
    sim_mat_list = []
    for z in z_list:
        sim_mat = cosine_similarity(z)
        sim_mat_list.append(sim_mat)

    if perm_type == 'random':
        rho_array, perm_list = permutation_analysis_random(
            sim_mat_list, max_perm=max_perm, n_known=n_known
        )
    elif perm_type == 'last':
        rho_array, perm_list = permutation_analysis_last(
            sim_mat_list, max_perm=max_perm, n_known=n_known
        )
    elif perm_type == 'sub':
        rho_array, perm_list = permutation_analysis_sub(
            sim_mat_list, max_perm=max_perm, n_unknown=n_unknown,
            is_exact=is_exact
        )
    else:
        raise ValueError(
            'Permutation analysis {0} is not implemented'.format(perm_type)
        )
    # Compute proportion of permutations worse (i.e., less) than the correct
    # alignment.
    n_perm = len(rho_array) - 1
    perm_percentile = np.sum(np.less(rho_array[1:], rho_array[0])) / n_perm
    return perm_percentile, rho_array, perm_list


def permutation_analysis_last(sim_mat_list, max_perm=10000, n_known=0):
    """Perform last-two permutation analysis.

    Swap same rows of matrices. If more than two matrices, must decide
    which matrices will have there rows swapped (although at least one
    will always have their rows swapped).

    In order to evaluate our ability to correctly align matrices, we
    first compute the correlation of the correct ordering and then
    compute the correlations for the sampled permutations. What we
    would like to see is that the correlation for the correct ordering
    is higher than any of the permuted alignments.
    """
    n_sim_mat = len(sim_mat_list)
    alignment_combos = list(itertools.combinations(np.arange(n_sim_mat, dtype=int), 2))

    n_item = sim_mat_list[1].shape[0]

    # Sample from possible permutations. We only consider permutations that
    # have a single pair of indices swapped. For ten elements there are
    # 45 potential combinations.
    list_swap_idx = list(
        itertools.combinations(np.arange(n_item - n_known, dtype=int), 2)
    )
    list_swap_idx = np.array(list_swap_idx, dtype=int)
    n_perm = np.minimum(max_perm, list_swap_idx.shape[0], dtype=int)
    # Select subset of swaps out of all possibilties.
    rand_idx = np.random.permutation(len(list_swap_idx))[0:n_perm]
    selected_swap_idx = list_swap_idx[rand_idx, :]

    # If more than two matrices, determine which matrices will have swapped
    # values. The first matrix is always not swapped, but there are many
    # other possibilities for remaining matrices (e.g., with three matrices:
    #  001, 010, or 011)
    if (n_sim_mat - 1) == 1:
        is_swapped = np.ones((n_perm, 1), dtype=int)
    else:
        # If more than two matrices, swap matrices stochastically with at least
        # one matrix being swapped.
        s = list(itertools.product(np.array([0, 1], dtype=int), repeat=n_sim_mat-1))
        # Drop case where no matrix has rows swapped.
        s = np.array(s[1:], dtype=int)
        n_outcome = s.shape[0]
        outcome_draw = np.random.randint(0, n_outcome, n_perm)
        is_swapped = s[outcome_draw, :]
        # Swap all matrices.
        # is_swapped = np.ones((n_perm, n_sim_mat - 1), dtype=int)

    # Flesh out matrices of indices detailing all permutations to be used.
    dmy_idx = np.arange(n_item, dtype=int)
    perm_list_all = []
    for i_sim_mat in range(n_sim_mat - 1):
        perm_list = np.tile(np.expand_dims(dmy_idx, axis=0), [n_perm + 1, 1])
        for i_perm in range(n_perm):
            if is_swapped[i_perm, i_sim_mat] == 1:
                old_value_0 = copy.copy(perm_list[i_perm + 1, selected_swap_idx[i_perm, 0]])
                old_value_1 = copy.copy(perm_list[i_perm + 1, selected_swap_idx[i_perm, 1]])
                perm_list[i_perm + 1, selected_swap_idx[i_perm, 0]] = old_value_1
                perm_list[i_perm + 1, selected_swap_idx[i_perm, 1]] = old_value_0
        perm_list_all.append(perm_list)

    # We store the correlation with correct ordering at the first idx, and the
    # correlations for the swapped ordering in the remaining indices.
    rho_array = np.zeros([n_perm + 1])
    # Correct alignment.
    rho_array[0] = alignment_score_multi(sim_mat_list, alignment_combos)
    # Permuted alignments.
    for perm_idx in range(n_perm):
        sim_mat_perm_list = []
        # Add unpermuted matrix.
        sim_mat_perm_list.append(sim_mat_list[0])
        # Add permuted matrices.
        for sim_mat_idx in range(n_sim_mat - 1):
            sim_mat_perm_list.append(
                symmetric_matrix_indexing(sim_mat_list[sim_mat_idx + 1], perm_list_all[sim_mat_idx][perm_idx + 1, :])
            )
        # Compute score
        rho_array[perm_idx + 1] = alignment_score_multi(
            sim_mat_perm_list, alignment_combos
        )
    return rho_array, perm_list_all


def permutation_analysis_random(sim_mat_list, max_perm=10000, n_known=0):
    """Perform random permutation analysis.

    In order to evaluate our ability to correctly align matrices, we
    first compute the correlation of the correct ordering and then
    compute the correlations for the sampled permutations. What we
    would like to see is that the correlation for the correct ordering
    is higher than any of the permuted alignments.
    """
    n_sim_mat = len(sim_mat_list)
    n_item = sim_mat_list[1].shape[0]
    n_perm = np.minimum(max_perm, math.factorial(n_item - n_known))
    alignment_combos = list(itertools.combinations(np.arange(n_sim_mat, dtype=int), 2))

    # Sample from possible permutations. There are 3,628,800 permutations of
    # ten elements (i.e., 10!), so it is very unlikely to get repeated samples.
    perm_list_all = []
    fixed_idx = np.arange(n_item - n_known, n_item)
    for _ in range(n_sim_mat - 1):
        perm_list = np.zeros([n_perm + 1, n_item], dtype=int)
        perm_list[0, :] = np.arange(n_item)
        for i_perm in range(n_perm):
            perm_list[i_perm + 1, 0:(n_item - n_known)] = np.random.permutation(n_item - n_known)
            perm_list[i_perm + 1, (n_item - n_known):] = fixed_idx
        perm_list_all.append(perm_list)

    # We store the correlation with correct ordering at the first idx, and the
    # correlations for the swapped ordering in the remaining indices.
    rho_array = np.zeros([n_perm + 1])
    # Correct alignment.
    rho_array[0] = alignment_score_multi(sim_mat_list, alignment_combos)
    # Permuted alignments.
    for perm_idx in range(n_perm):
        sim_mat_perm_list = []
        # Add unpermuted matrix.
        sim_mat_perm_list.append(sim_mat_list[0])
        # Add permuted matrices.
        for sim_mat_idx in range(n_sim_mat - 1):
            sim_mat_perm_list.append(
                symmetric_matrix_indexing(sim_mat_list[sim_mat_idx + 1], perm_list_all[sim_mat_idx][perm_idx + 1,:])
            )
        # Compute score
        rho_array[perm_idx + 1] = alignment_score_multi(
            sim_mat_perm_list, alignment_combos
        )
    return rho_array, perm_list_all


def permutation_analysis_sub(sim_mat_list, max_perm=10000, n_unknown=2, is_exact=False):
    """Perform random permutation analysis.

    In order to evaluate our ability to correctly align matrices, we
    first compute the correlation of the correct ordering and then
    compute the correlations for the sampled permutations. What we
    would like to see is that the correlation for the correct ordering
    is higher than any of the permuted alignments.
    """
    if is_exact:
        n_mismatch_thresh = 1
    else:
        n_mismatch_thresh = n_unknown

    n_sim_mat = len(sim_mat_list)
    n_item = sim_mat_list[0].shape[0]
    n_perm = np.minimum(max_perm, math.factorial(n_item))
    alignment_combos = list(itertools.combinations(np.arange(n_sim_mat, dtype=int), 2))
    ordered_idx = np.arange(n_item)
    # Initialize perm_list_all.
    perm_list_all = []
    for _ in range(n_sim_mat - 1):
        perm_list = np.zeros([n_perm + 1, n_item], dtype=int)
        perm_list[0, :] = np.arange(n_item)
        perm_list_all.append(perm_list)

    # Sample from possible permutations. There are 3,628,800 permutations of
    # ten elements (i.e., 10!), so it is very unlikely to get repeated samples.
    for i_perm in range(n_perm):
        # Randomly select indices that will be 'unknown'.
        sub_idx = np.random.permutation(n_item)
        sub_idx = sub_idx[0:n_unknown]
        sub_idx = np.sort(sub_idx)
        for i_sim_mat in range(n_sim_mat - 1):
            perm_idx = copy.copy(ordered_idx)
            aligned = True
            while aligned:
                # Permute indices selected as 'unknown'.
                perm_sub_idx = np.random.permutation(sub_idx)
                # Check number of mismatches.
                n_mismatch = np.sum(np.equal(sub_idx, perm_sub_idx))
                if i_sim_mat == 0:
                    # The first matrix must have at least the number of
                    # request mismatches.
                    if n_mismatch < n_mismatch_thresh:
                        aligned = False
                else:
                    # All other matrices don't matter.
                    aligned = False
            perm_idx[sub_idx] = perm_sub_idx
            perm_list_all[i_sim_mat][i_perm + 1, :] = perm_idx

    # We store the correlation with correct ordering at the first idx, and the
    # correlations for the swapped ordering in the remaining indices.
    rho_array = np.zeros([n_perm + 1])
    # Correct alignment.
    rho_array[0] = alignment_score_multi(sim_mat_list, alignment_combos)
    # Permuted alignments.
    for perm_idx in range(n_perm):
        sim_mat_perm_list = []
        # Add unpermuted matrix.
        sim_mat_perm_list.append(sim_mat_list[0])
        # Add permuted matrices.
        for sim_mat_idx in range(n_sim_mat - 1):
            sim_mat_perm_list.append(
                symmetric_matrix_indexing(sim_mat_list[sim_mat_idx + 1], perm_list_all[sim_mat_idx][perm_idx + 1, :])
            )
        # Compute score
        rho_array[perm_idx + 1] = alignment_score_multi(
            sim_mat_perm_list, alignment_combos
        )
    return rho_array, perm_list_all


def alignment_score_multi(sim_mat_list, alignment_combos):
    """"""
    score = 0
    weight = 1 / len(alignment_combos)
    for combo in alignment_combos:
        score = score + weight * alignment_score(
            sim_mat_list[combo[0]],
            sim_mat_list[combo[1]]
        )
    return score


def alignment_score(a, b, method='spearman'):
    """Return the alignment score between two similarity matrices.

    Assumes that matrix a is the smaller matrix and crops matrix b to
    be the same shape.
    """
    n_row = a.shape[0]
    b_cropped = b[0:n_row, :]
    b_cropped = b_cropped[:, 0:n_row]
    idx_upper = np.triu_indices(n_row, 1)

    if method == 'spearman':
        # Alignment score is the Spearman correlation coefficient.
        alignment_score, _ = spearmanr(a[idx_upper], b_cropped[idx_upper])
    else:
        raise ValueError(
            "The requested method '{0}'' is not implemented.".format(method)
        )
    return alignment_score


def symmetric_matrix_indexing(m, perm_idx):
    """Index matrix symmetrically.

    Can be used to symmetrically swap both rows and columns or to
    subsample.
    """
    m_perm = copy.copy(m)
    m_perm = m_perm[perm_idx, :]
    m_perm = m_perm[:, perm_idx]
    return m_perm


def infer_mistmatch(perm_list):
    """"""
    n_mismatch_list = []
    for perm_array in perm_list:
        (n_perm, n_class) = perm_array.shape
        n_mismatch = np.zeros((n_perm))
        true_idx = np.arange(n_class)
        for i_perm in range(n_perm):
            n_mismatch[i_perm] = np.sum(np.not_equal(true_idx, perm_array[i_perm]))
        n_mismatch_list.append(copy.copy(n_mismatch))
    return n_mismatch_list


def print_ttest(a, b):
    """"""
    df = len(a) - 1
    tstat, pval = ttest_ind(a, b)
    print('t({0})={1:.2f}, p={2:.3f}'.format(df, tstat, pval))


def build_intersect_path(dataset_list, fp_base, include_aoa=False):
    """"""
    fp_intersect = fp_base / Path('intersect')
    fn = 'intersect_'
    for i_dataset in dataset_list:
        fn = fn + i_dataset + '-'
    # Remove last dash.
    fn = fn[0:-1]
    if include_aoa:
        fn = fn + '-aoa'
    fn = fn + '.p'
    fp_intersect = fp_intersect / Path(fn)
    return fp_intersect


if __name__ == "__main__":
    is_demo = True
    # Change the path to point to the root of the project.
    fp_base = Path.home() / Path('Desktop\master')
    main(fp_base, is_demo)
