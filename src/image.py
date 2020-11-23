import cv2
import numpy as np

from tensorflow.keras.models import load_model


def cannyEdge(img, sigma=0.333):
    v = np.median(img)

    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    return cv2.Canny(img, lower, upper)


def angle_cos(p0, p1, p2):
    d1, d2 = (p0-p1).astype('float'), (p2-p1).astype('float')
    return abs( np.dot(d1, d2) / np.sqrt( np.dot(d1, d1)*np.dot(d2, d2) ) )


def find_squares(img):
    squares = []

    for thrs in range(0, 255, 26):
        if thrs == 0:
            # bin = cv2.Canny(img, 0, 50, apertureSize=5)
            bin = cannyEdge(img)
            bin = cv2.dilate(bin, None)
        else:
            _retval, bin = cv2.threshold(img, thrs, 255, cv2.THRESH_BINARY)
        contours, _hierarchy = cv2.findContours(bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            cnt_len = cv2.arcLength(cnt, True)
            cnt = cv2.approxPolyDP(cnt, 0.02*cnt_len, True)
            if len(cnt) == 4 and cv2.contourArea(cnt) > 1000 and cv2.isContourConvex(cnt):
                cnt = cnt.reshape(-1, 2)
                max_cos = np.max([angle_cos( cnt[i], cnt[(i+1) % 4], cnt[(i+2) % 4] ) for i in range(4)])
                if max_cos < 0.1:
                    squares.append(cnt)
    
    return squares


def in_range(ref, val, value_range=10):
    """ Check if the input values are within the defined value range """
    return (ref + value_range) >= val and (ref - value_range) <= val


def in_range_2d(ref, pnt, value_range=10):
    """ Check if the two given points are within the defined value range """
    return (
        in_range(ref[0], pnt[0], value_range=value_range)
        and in_range(ref[1], pnt[1], value_range=value_range)
    )


def threshold(image):
    return cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 111, 2)


def opening(image, iterations=1, kernel=(3, 3)):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel)
    image = cv2.erode(image, kernel, iterations=iterations)
    return cv2.dilate(image, kernel, iterations=iterations)


def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


def show_image(image):
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def sharpenImage(image):
    """ Use a kernel filter for sharpen an image """
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    return cv2.filter2D(image, -1, kernel)


class NoBoardFound(Exception):
    pass


class SudokuImage:

    # TODO: Make this take raw file like data as well
    def __init__(self, image, dim=(512, 512)):
        if isinstance(image, str):
            image = cv2.imread(image, cv2.IMREAD_COLOR)
        self.source_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
        self.grey_image = cv2.cvtColor(self.source_image, cv2.COLOR_BGR2GRAY)

    def find(self):
        try:
            self.find_board()
            self.find_cells()
            assert len(self.distinct_cells) == 81
            self.order_cells()
            self.identify_cells()
        except (NoBoardFound, AssertionError) as err:
            self.board = None

    @property
    def max_cell_area(self):
        """ Maximum possible size of a sudoku cell for the image """
        height, width = self.grey_image.shape
        approx_height = height // 9
        approx_width = width // 9
        return approx_height * approx_width

    @staticmethod
    def is_contour_square(contour):
        # a square will have an aspect ratio that is approximately
        # equal to one, otherwise, the shape is a rectangle
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h)
        return aspect_ratio >= 0.95 and aspect_ratio <= 1.05

    def find_board(self):
        image = cv2.GaussianBlur(self.grey_image, (7,7), 0)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        image = sharpenImage(image)
        image = cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 111, 2)
        squares = find_squares(image)

        squares = [cv2.boundingRect(contour) for contour in squares
                   if cv2.contourArea(contour) > self.max_cell_area and self.is_contour_square(contour)]
        squares = sorted(squares, key=lambda obj: obj[2] * obj[3])

        try:
            board = squares[0]
            x, y, w, h = board
        except IndexError:
            raise NoBoardFound()

        self.board_offset = (x, y)
        self.sudoku_board = image[y:y+h, x:x+w]
        self.grey_board = self.grey_image[y:y+h, x:x+w]

    def find_cells(self):
        squares = find_squares(self.sudoku_board)
        squares = [
            cv2.boundingRect(contour) for contour in squares
            if cv2.contourArea(contour) < self.max_cell_area
        ]

        self.distinct_cells = []
        distinct_corners = []

        # Filter out duplicates by comparing how close the top left corners are
        for square in squares:
            x, y, w, h = square
            if not any((in_range_2d(ref, (x, y)) for ref in distinct_corners)):
                distinct_corners.append((x, y))
                self.distinct_cells.append(square)

    def order_cells(self):
        by_height = sorted(self.distinct_cells, key=lambda x: x[1])
        self.distinct_cells = []
        for i in range(0, 81, 9):
            self.distinct_cells += sorted(by_height[i:i+9], key=lambda x: x[0])

    def identify_cells(self):
        # Load trained MNIST model for guessing digits
        model = load_model('digits_model.h5')
        self.board = ''
        for square in self.distinct_cells:
            x, y, w, h = square
            ow = int(w / 10)
            oh = int(h / 10)
            x += ow
            w -= 2 * ow
            y += oh
            h -= 2 * oh

            # Preprocess each cell image for number detection
            cell_image = self.sudoku_board[y:y + h, x:x + w]
            image = self.grey_board[y:y + h, x:x + w]
            if np.mean(cell_image) < 250:
                image = cv2.GaussianBlur(image, (3, 3), 0)
                image = sharpenImage(image)
                image = thresholding(image)
                image = ~image
                # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                # image = cv2.dilate(image, kernel, iterations=1)
                # image = cv2.erode(image, kernel, iterations=1)

                # Set the cell image to a format that the model will understand
                image = cv2.resize(image, (28, 28), interpolation=cv2.INTER_AREA)
                image = image.reshape((1, 28, 28, 1))
                image = image.astype(dtype=np.float64)
                image /= 255

                prediction = model.predict(image)
                self.board += str(prediction.argmax())
            else:
                self.board += '0'

    def board_to_image(self):
        xo, yo = self.board_offset
        for square, value in zip(self.distinct_cells, self.board):
            x, y, w, h = square
            y += yo
            x += xo
            self.source_image = cv2.putText(
                self.source_image, str(value),
                (x, y + h), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 255, 0) , 2, cv2.LINE_AA)
