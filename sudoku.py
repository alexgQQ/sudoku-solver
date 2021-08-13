import os
import click
import glob
import cv2
import random
import functools
import random
import csv
import numpy as np

from collections import defaultdict
from multiprocessing import Pool, cpu_count

from tensorflow.keras.models import load_model
from src import SudokuImage, solve_sudoku, array_to_str, str_to_array, train_model, show_image


def get_rows():
    with open('sudoku.csv', 'r', newline='') as src:
        reader = csv.reader(src)
        # Skip csv headers
        next(reader, None)
        for row in reader:
            yield row


@click.group()
@click.pass_context
def cli(ctx):
    '''
    Main entrypoint into the application
    '''
    ctx.ensure_object(dict)


@cli.command()
@click.argument(
    'board', required=True, type=str)
@click.option(
    '--output', default='stdout', type=click.Choice(['image', 'string', 'stdout']),
    show_default=True, help='Output format of solution')
@click.pass_context
def solve(ctx, board, output):
    '''
    Solve a given sudoku board.

    BOARD - This can be a filepath to an image of a sudoku board or
    a string of 81 numbers of a sudoku board in order from left to right
    '''
    try:
        is_file = False
        if any([board.endswith(file_extension) for file_extension in ['.jpg', 'jpeg', '.png']]):
            path = os.path.abspath(board)
            if not os.path.exists(path):
                raise TypeError
            is_file = True
            click.secho(f'Using image file {path} ...', fg='cyan')
            sudoku = SudokuImage(path)
            sudoku.find()
            board = sudoku.board
            assert board is not None
        board = str_to_array(board)
    except TypeError:
        click.secho(f'File {path} does not exist', err=True, fg='red')
        return
    except ValueError as err:
        click.secho(f'Invalid board values, should be 9x9 in size - {err}', err=True, fg='red')
        return
    except AssertionError:
        click.secho(f'Unable to find sudoku board from image', err=True, fg='red')
        return

    click.secho(f'Finding solution...', fg='cyan')

    try:
        solution = solve_sudoku(board)
    except Exception as err:
        solution = None
        click.secho(f'Unable to find solution, this usually means the board is incorrect', err=True, fg='red')

    if output == 'string':
        solution = array_to_str(solution)
        click.secho(solution, fg='green')
        return

    elif is_file and output == 'image':
        if solution is not None:
            sudoku.board = array_to_str(solution)
        sudoku.board_to_image()
        show_image(sudoku.source_image)

    else:
        board_string = '\n-----------\n'
        for index, value in enumerate(board.flatten()):
            index += 1
            board_string += str(value)
            if index % 27 == 0:
                board_string += '\n-----------\n'
            elif index % 9 == 0:
                board_string += '\n'
            elif index % 3 == 0:
                board_string += '|'
        click.secho(f'{board_string}', fg='green')


@cli.command()
@click.pass_context
def train(ctx):
    train_model()


@cli.command()
@click.option(
    '-c', '--component', default='all', type=click.Choice(['all', 'solver', 'images', 'model']),
    show_default=True, help='Which piece of the sudoku solver to test')
@click.pass_context
def test(ctx, component):

    if component in ('solver', 'all'):
        for row in get_rows():
            board, truth = row
            board = str_to_array(board)
            solution = solve_sudoku(board)
            solution = array_to_str(solution)
            assert solution == truth

    if component in ('images', 'all'):

        image_paths = glob.glob('images/*.jpg')

        # Data structs to track testing data
        guesses = defaultdict(int)
        truths = defaultdict(int)

        total_images = 0
        boards_found = []
        total_guesses = 0
        wrong_guesses = 0
        total_correct = 0

        found_boards = []
        for image_path in image_paths:
            total_images += 1
            image = SudokuImage(image_path)
            image.find_board()

            data_file_path = image_path.replace('.jpg', '.dat')
            with open(data_file_path, 'r') as fp:
                lines = fp.readlines()
            valid = ''.join([line.replace(' ', '').replace('\n', '') for line in lines[2:]])

            if image.board is not None:
                found_boards.append((image, valid))

        click.secho(f'Total Images: {total_images}')
        click.secho(f'Large Squares Found: {len(found_boards)}')

        full_boards = []
        for sudoku, valid in found_boards:
            sudoku.find_cells()
            if len(sudoku.distinct_cells) == 81:
                sudoku.order_cells()
                full_boards.append((sudoku, valid))

        click.secho(f'All Cells Found: {len(full_boards)}')

        correct = []
        boards_found = []
        for sudoku, valid in full_boards:
            sudoku.identify_cells()

            valid_arr = str_to_array(valid).flatten()
            guess_arr = str_to_array(sudoku.numbers).flatten()
            boards_found.append(
                (sudoku.image_path, np.linalg.norm(valid_arr - guess_arr))
            )

            if sudoku.numbers == valid:
                total_correct += 1

            # See how many of the guesses were incorrect
            for guess, truth in zip(sudoku.numbers, valid):
                if truth != '0':
                    total_guesses += 1
                if truth != guess:
                    wrong_guesses += 1
                    truths[truth] += 1
                    guesses[guess] += 1

        # Calculate the percentage of each digit that was wrong
        for key, val in truths.items():
            truths[key] = val / wrong_guesses

        for key, val in guesses.items():
            guesses[key] = val / wrong_guesses


        click.secho(f'- {sorted(boards_found, key=lambda x: x[1])}')
        click.secho(f'Total Guesses: {total_guesses}')
        click.secho(f'Total Correct: {total_correct}')
        click.secho(f'% Wrong Guesses: {wrong_guesses/total_guesses}')
        click.secho(f'% Truths Wrong: {truths}')
        click.secho(f'% Guesses Wrong: {guesses}')


if __name__ == '__main__':

    cli()

