import matplotlib.pyplot as plt
import json
import os
import shutil
import numpy as np



if __name__=="__main__":

    detectron_scores=json.load(open("../background_removal_module/output_metrics.json","r"))

    # set width of bar
    barWidth = 0.25
    fig = plt.subplots(figsize=(12, 8))

    # set height of bar
    IOU = [detectron_scores["iou_mean"], 30, 1]
    DICE = [detectron_scores["dice_mean"],None, 16]
    #CSE = [29, 3, 24, 25, 17]

    # Set position of bar on X axis
    br1 = np.arange(len(IOU))
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]

    # Make the plot
    plt.bar(br1, IOU, color='r', width=barWidth,
            edgecolor='grey', label='IOU')
    plt.bar(br2, DICE, color='g', width=barWidth,
            edgecolor='grey', label='DICE')

    # Adding Xticks
    plt.xlabel('Score', fontweight='bold', fontsize=15)
    plt.ylabel('Models', fontweight='bold', fontsize=15)
    plt.xticks([r + barWidth for r in range(len(IOU))],
               ['Detectron2', 'Detectron2+Median', 'Unet'])

    plt.legend()
    plt.show()