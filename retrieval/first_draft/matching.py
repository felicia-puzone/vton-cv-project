import cv2 as cv
import numpy as np
import sys
import matplotlib.pyplot as plt
import builder
import pre_processing as pre
import logging as lgg

logger = lgg.getLogger(__name__)
FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
lgg.basicConfig(format=FORMAT)
logger.setLevel(lgg.DEBUG)


def match_n_best(query_img, repo_file='data_repo.pickle', n_best=5, ratio_match=0.1, verbose=False):
    logger.info("Loading repository...")
    repo_dict = builder.repository_loader(repo_file)

    orb = cv.ORB_create()
    query_kp, query_des = orb.detectAndCompute(query_img, None)

    logger.info("Processing repository...")
    matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

    # best_n_match -> dict (key: image name, value: score)
    best_n_img = []
    for item_key, (item_kp, item_des) in repo_dict.items():
        if item_des is None:
            logger.info(f"\tImage {item_key},\tNo descriptor available, so no possible comparison.")
            continue
        matches = matcher.match(query_des, item_des)
        matches = sorted(matches, key=lambda x: x.distance)

        # end_index modules how many matches are used to compute the score
        end_index = int(len(matches) * ratio_match)
        if end_index < 1:
            logger.info(f"\tImage {item_key},\tNo match at all")
            continue

        top_matches = matches[:end_index]
        penalties = []
        query_indices = [match.queryIdx for match in top_matches]
        train_indices = [match.trainIdx for match in top_matches]

        query_keypoints = [query_kp[idx] for idx in query_indices]
        train_keypoints = [item_kp[idx] for idx in train_indices]
        for match in top_matches:
            query_keypoint = query_kp[match.queryIdx]
            xq, yq = query_keypoint.pt
            train_keypoint = item_kp[match.trainIdx]
            xt, yt = train_keypoint.pt
            penalties.append(np.power(xq - xt, 2) + np.power(yq - yt, 2))

        # score is the mean square error computed on distances between the matches
        distances = [match.distance for match in top_matches]
        if len(distances) > 0:
            match_score = np.sum(np.power(distances, 2)) / len(distances)
            penalty_score = np.log(np.sum(penalties))
            score = (match_score / penalty_score)
        else:
            score = 2000

        if verbose:
            logger.info(f"\tImage {item_key}, \tscore: {score}")

        best_n_img.append((item_key, score))
        best_n_img.sort(key=lambda x: x[1])
        if len(best_n_img) > n_best:
            best_n_img.pop()

    return best_n_img


def plot_match_n_best(query_img, best_n_img, ratio_match=0.1, repo_path='./data/', repo_file='data_repo.pickle'):
    repo_dict = builder.repository_loader(repo_file)

    orb = cv.ORB_create()
    bf = cv.BFMatcher()

    query_kp, query_des = orb.detectAndCompute(query_img, None)

    for img_name, img_score in best_n_img:
        matches = bf.knnMatch(query_des, repo_dict[img_name][1], k=2)
        best_matches = [m1 if m1.distance > m2.distance else m2 for m1, m2 in matches]
        best_matches = sorted(best_matches, key=lambda x: x.distance)
        end_index = int(len(best_matches) * ratio_match)
        ref_img = cv.imread(repo_path + img_name, cv.IMREAD_GRAYSCALE)
        img3 = cv.drawMatches(query_img, query_kp, ref_img, repo_dict[img_name][0], best_matches[:end_index], None,
                          flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        plt.imshow(img3), plt.show()


def match_draft_with_plot(query_img, repo_file='data_repo.pickle', img_path='./data/', verbose=False):
    print("Loading repository...")
    repo_dict = builder.repository_loader(repo_file)

    orb = cv.ORB_create()
    query_kp, query_des = orb.detectAndCompute(query_img, None)

    print("Processing repository...")
    # bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    bf = cv.BFMatcher()

    best_score = sys.float_info.max
    best_img_name = None
    for item_key, (item_kp, item_des) in repo_dict.items():
        if item_des is None:
            print("\timage '", item_key, "'\tNo descriptor available, so no possible comparison.")
            continue
        matches = bf.knnMatch(query_des, item_des, k=2)
        distances = np.array([m1.distance if m1.distance < m2.distance else m2.distance for m1, m2 in matches])
        distances = np.sort(distances)

        # end_index modules how many matches are used to compute the score
        end_index = int(len(distances) * 0.1)
        if end_index < 1:
            print("\timage '", item_key, "'\tNo match at all")
            continue

        # score is the mean square error computed on distances between the matches
        score = np.sum(np.power(distances[:end_index], 2)) / end_index

        if verbose:
            print("\timage '", item_key, "'\tscore:", score)

        if score < best_score:
            best_score = score
            best_img_name = item_key

    query_img_kp = cv.drawKeypoints(query_img, query_kp, None, color=(0, 255, 0), flags=0)
    ref_img = cv.imread('./data/' + best_img_name, cv.IMREAD_GRAYSCALE)
    ref_img_kp = cv.drawKeypoints(ref_img, repo_dict[best_img_name][0], None, color=(0, 255, 0), flags=0)
    img_kp = cv.hconcat([query_img_kp, ref_img_kp])
    plt.imshow(img_kp), plt.show()

    matches = bf.knnMatch(query_des, repo_dict[best_img_name][1], k=2)
    best_matches = [m1 if m1.distance > m2.distance else m2 for m1, m2 in matches]
    best_matches = sorted(best_matches, key=lambda x: x.distance)
    img3 = cv.drawMatches(query_img, query_kp, ref_img, repo_dict[best_img_name][0], best_matches[:end_index], None,
                          flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(img3), plt.show()

    return best_img_name, best_score


def plot_match(img_path_1, img_path_2):
    img_1 = cv.imread(img_path_1, cv.IMREAD_GRAYSCALE)
    img_2 = cv.imread(img_path_2, cv.IMREAD_GRAYSCALE)

    img_1 = pre.perform_pre_processing(img_1)

    orb = cv.ORB_create()
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

    kp_1, des_1 = orb.detectAndCompute(img_1, None)
    kp_2, des_2 = orb.detectAndCompute(img_2, None)

    #img3 = cv.hconcat([img_1, img_2])
    #plt.imshow(img3), plt.show()

    img1_kp = cv.drawKeypoints(img_1, kp_1, None, color=(0, 255, 0), flags=0)
    plt.imshow(img1_kp), plt.show()
    img2_kp = cv.drawKeypoints(img_2, kp_2, None, color=(0, 255, 0), flags=0)
    plt.imshow(img2_kp), plt.show()
    img_kp = cv.hconcat([img1_kp, img2_kp])
    plt.imshow(img_kp), plt.show()

    matches = bf.match(des_1, des_2)
    matches = sorted(matches, key=lambda x: x.distance)
    img3 = cv.drawMatches(img_1, kp_1, img_2, kp_2, matches[:100], None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    plt.imshow(img3), plt.show()
