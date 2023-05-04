import matplotlib.pyplot as plt
import json
import os
import shutil
import numpy as np



if __name__=="__main__":

    detectron_scores=json.load(open("../background_removal_module/output_metrics.json","r"))
    unet_scores=json.load(open("../background_removal_module/unet_scores.json","r"))
    detectron_gc_scores=json.load(open("../background_removal_module/detectron2+gc_scores.json","r"))
    # set width of bar
    barWidth = 0.25
    fig = plt.subplots(figsize=(12, 8))

    # set height of bar
    IOU = [detectron_scores["iou_mean"] ,detectron_gc_scores["iou_mean"], unet_scores["iou_mean"]]
    DICE = [detectron_scores["dice_mean"],detectron_gc_scores["dice_mean"], unet_scores["dice_mean"]]
    #CSE = [29, 3, 24, 25, 17]

    # Set position of bar on X axis
    br1 = np.arange(len(IOU))
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]

    # Make the plot
    plt.bar(br1, IOU, color='cyan', width=barWidth,
            edgecolor='gray', label='IOU')
    plt.bar(br2, DICE, color='pink', width=barWidth,
            edgecolor='gray', label='DICE')

    # Adding Xticks
    plt.xlabel('Models', fontweight='bold', fontsize=15)
    plt.ylabel('Scores', fontweight='bold', fontsize=15)
    print([r + barWidth for r in range(len(IOU))])
    plt.xticks([r + barWidth for r in range(len(IOU))],
               ['Detectron2',"Detectron2+GC+MedianFilter", 'Unet'])

    plt.legend()
    plt.show()