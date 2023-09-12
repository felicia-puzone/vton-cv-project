Retrieval module.

Training pipeline:
1. Get dataset ("in-shop" clothes; person-wearing-cloth)
2. Extract ORB keypoints and corresponding descriptors from in-shop cloth
3. Build feature vector from extracted descriptors for each cloth
4. Extract ORB keypoints and corresponding descriptors from person-wearing-cloth
3. Build feature vector from extracted descriptors for each person-wearing-cloth
6. Save feature vectors to repository
7. Train SimilarityNet

Inference pipeline:
1. Get cloth-worn-image
2. Extract ORB keypoints and corresponding descriptors from cloth-worn-image
3. Build feature vector from extracted descriptors
4. For each in-shop cloth:
      - Load in-shop cloth feature vector
      - Feed to SimilarityNet (stack of linear layers)
      - Rank the score-match
6. Get top-k-matches from the "in-shop" repository of clothes


Feature vector:
- concatenation of the keypoints' descriptors for each channel + histogram for each channel (!!! assuming rgb images !!!)
