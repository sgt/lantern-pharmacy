# mser might be better at finding blobs than findContours. or not?

import cv2

IMAGE = 'c:/users/sgt/work/lantern-pharmacy/test/data/pale.jpg'

if __name__ == '__main__':
    img = cv2.imread(IMAGE, cv2.IMREAD_GRAYSCALE)
    # _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh = img
    vis = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    mser = cv2.MSER_create(_min_area=10)  # _max_area=1000, _area_threshold=0)
    regions, bboxes = mser.detectRegions(thresh)
    print(f"total boxes: {len(bboxes)}")
    # hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions[0]]
    # cv2.polylines(vis, hulls, 1, (0, 255))
    for x, y, w, h in bboxes:
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 0, 255))
    cv2.imshow('img', vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
