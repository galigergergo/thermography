#!/usr/bin/env python
# coding: utf-8

# In[1]:

import os

# used to replace parts of the mask file name in order to yield the image file name
def getHybridPrefixes():
    return ["V_true_", "T_virt_abel_"]


# used to replace parts of the mask file name in order to yield the image file name
def getE2EPrefixes():
    return ["V_true_", "measurements_"]


# used to replace parts of the mask file name in order to yield the image file name
def getFKMIGPrefixes():
    return ["V_true_", "fkmig_"]


# used to replace parts of the mask file name in order to yield the image file name
def getTSAFTPrefixes():
    return ["V_true_", "tsaft_"]


# used to replace parts of the mask file name in order to yield the image file name
def getRegTSAFTPrefixes():
    return ["V_true_", "V_rec_tsaft_"]


# locations of the pretrained compact models for the hybrid approach as used in the paper
def get_cmp_hybrid_models():
    cwd = os.getcwd()

    prefix = os.path.join(cwd, 'data', 'hybrid', 'models', 'cmp')
    postfix = os.path.join('models', 'best_model.pth')
    models = [
        #os.path.join(prefix, '10k', '2020-4-25_11-1-21.747849', postfix),
        #os.path.join(prefix, '10k', '2020-5-6_10-21-7.29690', postfix),
        #os.path.join(prefix, '10k', '2020-5-6_15-54-51.980621', postfix),
        #os.path.join(prefix, '10k', '2020-5-6_21-1-50.73327', postfix),
        #os.path.join(prefix, '10k', '2020-5-9_12-49-44.293454', postfix),
        #os.path.join(prefix, '20k', '2020-4-25_12-19-38.862970', postfix),
        #os.path.join(prefix, '20k', '2020-5-6_10-20-50.995568', postfix),
        #os.path.join(prefix, '20k', '2020-5-6_17-55-23.466756', postfix),
        #os.path.join(prefix, '20k', '2020-5-7_8-1-58.671300', postfix),
        #os.path.join(prefix, '20k', '2020-5-7_8-2-18.292168', postfix),
        #os.path.join(prefix, '40k', '2020-4-25_12-20-17.282749', postfix),
        #os.path.join(prefix, '40k', '2020-5-6_10-18-41.486786', postfix),
        #os.path.join(prefix, '40k', '2020-5-6_21-2-2.514594', postfix),
        #os.path.join(prefix, '40k', '2020-5-7_8-2-43.185166', postfix),
        #os.path.join(prefix, '40k', '2020-5-7_8-3-14.704469', postfix),
        #os.path.join(prefix, '80k', '2020-4-25_13-32-31.889297', postfix)
        #os.path.join(prefix, '80k', '2020-5-6_10-35-14.731866', postfix),
        #os.path.join(prefix, '80k', '2020-5-7_9-5-53.337821', postfix),
        #os.path.join(prefix, '80k', '2020-5-7_9-6-10.792490', postfix),
        #os.path.join(prefix, '80k', '2020-5-7_9-11-38.104847', postfix),
        os.path.join(prefix, '80k', 'only_square_admm', '2020-4-25_13-32-31.889297', postfix)
        #os.path.join(prefix, '80k', 'more_shapes_admm', '2022-12-23_23-22-19.858192', postfix)
#         os.path.join(prefix, '80k', 'more_shapes_fistanet', '2022-12-23_8-15-47.801755', postfix)
    ]

    return models


# locations of the pretrained large models for the hybrid approach as used in the paper
def get_lrg_hybrid_models():
    cwd = os.getcwd()

    prefix = os.path.join(cwd, 'data', 'hybrid', 'models', 'lrg')
    postfix = os.path.join('models', 'best_model.pth')
    models = [
        #os.path.join(prefix, '10k', '2020-4-26_9-53-31.62874', postfix),
        #os.path.join(prefix, '10k', '2020-4-26_22-12-23.766958', postfix),
        #os.path.join(prefix, '10k', '2020-4-27_19-5-34.18363', postfix),
        #os.path.join(prefix, '10k', '2020-5-6_10-29-8.417074', postfix),
        #os.path.join(prefix, '10k', '2020-5-6_10-29-21.776337', postfix),
        #os.path.join(prefix, '20k', '2020-4-26_22-11-2.94991', postfix),
        #os.path.join(prefix, '20k', '2020-5-6_10-28-44.486552', postfix),
        #os.path.join(prefix, '20k', '2020-5-6_10-29-33.100065', postfix),
        #os.path.join(prefix, '20k', '2020-5-6_18-57-44.934634', postfix),
        #os.path.join(prefix, '20k', '2020-5-6_18-57-59.270992', postfix),
        #os.path.join(prefix, '40k', '2020-4-26_9-52-39.366893', postfix),
        #os.path.join(prefix, '40k', '2020-4-27_19-5-47.521265', postfix),
        #os.path.join(prefix, '40k', '2020-5-1_8-52-48.343962', postfix),
        #os.path.join(prefix, '40k', '2020-5-2_11-4-55.392596', postfix),
        #os.path.join(prefix, '40k', '2020-5-3_19-5-2.128281', postfix),
        #os.path.join(prefix, '80k', '2020-4-25_15-5-29.503889', postfix)
        #os.path.join(prefix, '80k', '2020-4-26_22-10-31.830771', postfix),
        #os.path.join(prefix, '80k', '2020-4-27_19-3-23.42861', postfix),
        #os.path.join(prefix, '80k', '2020-5-1_8-50-27.682980', postfix),
        #os.path.join(prefix, '80k', '2020-5-3_19-5-23.801872', postfix),
        # os.path.join(prefix, '80k', 'only_square_admm', '2020-4-25_15-5-29.503889', postfix)
        os.path.join(prefix, '80k', 'only_square_admm', postfix)
#         os.path.join(prefix, '80k', 'more_shapes_fistanet', postfix)
    ]

    return models


# locations of the pretrained compact models for the end-to-end approach as used in the paper
def get_cmp_e2e_models():
    cwd = os.getcwd()

    prefix = os.path.join(cwd, 'data', 'end2end', 'models', 'cmp')
    postfix = os.path.join('models', 'best_model.pth')
    models = [
        os.path.join(prefix, '10k', '2020-4-17_13-11-56.59708', postfix),
        os.path.join(prefix, '10k', '2020-4-18_21-40-26.40950', postfix),
        os.path.join(prefix, '10k', '2020-4-21_9-37-5.815455', postfix),
        os.path.join(prefix, '10k', '2020-4-29_17-3-24.501940', postfix),
        os.path.join(prefix, '10k', '2020-5-9_12-53-30.347826', postfix),
        os.path.join(prefix, '20k', '2020-4-17_13-16-25.77976', postfix),
        os.path.join(prefix, '20k', '2020-4-18_21-40-35.787294', postfix),
        os.path.join(prefix, '20k', '2020-4-21_9-37-11.599037', postfix),
        os.path.join(prefix, '20k', '2020-4-29_17-3-19.547649', postfix),
        os.path.join(prefix, '20k', '2020-5-9_12-53-11.557119', postfix),
        os.path.join(prefix, '40k', '2020-4-17_13-17-3.636752', postfix),
        os.path.join(prefix, '40k', '2020-4-18_21-40-43.101733', postfix),
        os.path.join(prefix, '40k', '2020-4-21_9-37-19.295457', postfix),
        os.path.join(prefix, '40k', '2020-4-29_17-3-13.585412', postfix),
        os.path.join(prefix, '40k', '2020-5-2_16-51-49.997930', postfix),
        os.path.join(prefix, '80k', '2020-4-17_13-17-24.661566', postfix),
        os.path.join(prefix, '80k', '2020-4-18_21-40-51.57509', postfix),
        os.path.join(prefix, '80k', '2020-4-21_9-37-25.914759', postfix),
        os.path.join(prefix, '80k', '2020-4-29_17-2-33.818002', postfix),
        os.path.join(prefix, '80k', '2020-5-2_16-51-4.97723', postfix),
    ]

    return models


# locations of the pretrained large models for the end-to-end approach as used in the paper
def get_lrg_e2e_models():
    cwd = os.getcwd()

    prefix = os.path.join(cwd, 'data', 'end2end', 'models', 'lrg')
    postfix = os.path.join('models', 'best_model.pth')
    models = [
        os.path.join(prefix, '10k', '2020-4-17_11-14-58.105151', postfix),
        os.path.join(prefix, '10k', '2020-4-18_21-39-33.2678', postfix),
        os.path.join(prefix, '10k', '2020-4-21_9-36-41.699852', postfix),
        os.path.join(prefix, '10k', '2020-4-28_13-9-56.237617', postfix),
        os.path.join(prefix, '10k', '2020-4-29_16-56-49.715415', postfix),
        os.path.join(prefix, '20k', '2020-4-17_12-56-14.770435', postfix),
        os.path.join(prefix, '20k', '2020-4-18_21-39-59.905029', postfix),
        os.path.join(prefix, '20k', '2020-4-21_9-36-48.101709', postfix),
        os.path.join(prefix, '20k', '2020-4-28_13-11-12.51656', postfix),
        os.path.join(prefix, '20k', '2020-4-29_16-56-41.543742', postfix),
        os.path.join(prefix, '40k', '2020-4-17_13-0-19.68754', postfix),
        os.path.join(prefix, '40k', '2020-4-18_21-40-12.153832', postfix),
        os.path.join(prefix, '40k', '2020-4-21_9-36-53.962150', postfix),
        os.path.join(prefix, '40k', '2020-4-28_13-11-22.669288', postfix),
        os.path.join(prefix, '40k', '2020-4-29_16-56-34.151455', postfix),
        os.path.join(prefix, '80k', '2020-4-17_13-0-34.508471', postfix),
        os.path.join(prefix, '80k', '2020-4-18_21-40-19.367774', postfix),
        os.path.join(prefix, '80k', '2020-4-21_9-36-59.322816', postfix),
        os.path.join(prefix, '80k', '2020-4-28_13-11-29.803405', postfix),
        os.path.join(prefix, '80k', '2020-4-29_16-56-18.489670', postfix),
    ]

    return models
