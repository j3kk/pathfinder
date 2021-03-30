# q-learning process should take < 30s, game initialisation < 15s; q - quit, i - slow, o - medium, p - fast

import pandas as pd
import numpy as np
import pygame
import random

# ------------------------------ IMPORTING DATA ------------------------------
# Q-learning data: reward values (goal complete = 15, non-goal action = -1, barrier cross = -50, out of bounds = -100)
df_reward_values = pd.read_csv('pf_reward_values.csv', sep=',', header=None)

# Q-learning data: indices of next state after performing action (0 = hit, 1 = up, 2 = right, 3 = down, 4 = left)
df_next_state = pd.read_csv('pf_next_state_indices.csv', sep=',', header=None)

# Q-learning data: whether goal has been achieved (0 = no, 1 = yes) {NB: could probably have data in one csv file}
df_goal_achieved = pd.read_csv('pf_goal_achieved.csv', sep=',', header=None)

# Game data: xy coordinates of seeker and target
df_xy_coordinates = pd.read_csv('pf_xy_coordinates.csv', sep=',', header=None)

# ------------------------------ Q-LEARNING FUNCTION ------------------------------
# initialise a 256x5 (state x action) array with all cells set to 0 to serve as starting q-value table
q_values = np.array(np.zeros([256, 5]))


def q_pathfinder(iterations):

    alpha, gamma, epsilon = 0.8, 0.75, 0.9  # hyper-parameters: learning rate, discount factor, exploration/exploitation

    for i in range(1, iterations):  # main loop to calculate and update q-values

        if i % 5000 == 0:
            epsilon += - 0.1  # lowers epsilon value every 5000 iterations to reduce exploration over time

        current_state = random.randint(0, 255)  # randomly picks an index corresponding to a possible game state
        goal_achieved = 0  # value corresponds to whether goal has been achieved (0 = no, 1 - yes)

        while goal_achieved == 0:  # while loop broken when goal-achieving action is selected

            if random.uniform(0, 1) < epsilon:
                action = random.randint(0, 4)  # exploration: selects random action
            else:
                action = np.ndarray.argmax(q_values[current_state])  # exploitation: selects highest q-value action

            goal_achieved = df_goal_achieved.loc[current_state, action]  # value signifies whether action achieves goal
            reward_value = df_reward_values.loc[current_state, action]  # reward value of action: R(s,a)
            next_state = df_next_state.loc[current_state, action]  # index of next state after performing action: s'
            previous_q_value = q_values[current_state, action]  # q-value of action prior to this instance: |t-1| q(s,a)
            next_state_max_q = np.max(q_values[next_state])  # highest q-value possible at next state: |max a'| q(s',a')

            # calculates q-value of action (Bellman equation) and updates table with new value
            new_q_value = (1 - alpha) * previous_q_value + alpha * (reward_value + gamma * next_state_max_q)
            q_values[current_state, action] = new_q_value

            current_state = next_state  # set next iteration's starting state as the state reached by performing a


q_iterations = 25000  # default number of iterations for q-learning function
q_pathfinder(q_iterations)  # calls q-learning function prior to start of game

# ------------------------------ INITIALISING PATHFINDER ------------------------------
pygame.init()  # initialise game

# set game window dimensions and caption
window_width = 500
window_height = 500
window = pygame.display.set_mode((window_width, window_height))
pygame.display.update()
pygame.display.set_caption('Pathfinder')

# colour palette, clock and dimensions of seeker, target and movement
white, black, green, blue, silver = (255, 255, 255), (0, 0, 0), (0, 255, 0), (0, 255, 255), (192, 192, 192)
clock = pygame.time.Clock()
seeker_size, target_size, move_distance = 60, 80, 120


# ------------------------------ PATHFINDER GAME FUNCTIONS ------------------------------
# function to display text in game window
def display_text(text, colour, x_pos, y_pos, size):
    font = pygame.font.SysFont("freesansbold.ttf", size)
    text_to_show = font.render(text, True, colour)
    window.blit(text_to_show, [x_pos, y_pos])


# function returns random state with corresponding xy coordinates for seeker and target
def xy_generator():
    random_state = random.randint(0, 255)
    xy_coordinates = df_xy_coordinates.loc[random_state][0:5]
    return xy_coordinates


# function returns action with highest q-value in current state and corresponding index of next state
def optimal_route(current_state):
    best_action = np.ndarray.argmax(q_values[current_state])
    next_state = df_next_state.loc[current_state, best_action]
    return best_action, next_state


# ------------------------------ PATHFINDER GAME LOOP ------------------------------
def pathfinder_game():

    game_quit = False  # game closed whenever set to True
    target_hit = False  # True when best action at current state will achieve goal {NB: sure this could be cleaner}

    # variables reflect states, xy coordinates of seeker/target and present state's best (highest q-value) action
    current_state, seeker_x, seeker_y, target_x, target_y = xy_generator()
    optimal_action, next_state = optimal_route(current_state)

    # score/move/timekeeping
    score = 0
    moves = 0
    moves_this_level = 0
    seconds = 0
    time_since_last_update = 0
    time_interval = 1000

    while not game_quit:

        window.fill(silver)  # sets background colour
        pygame.draw.rect(window, white, [target_x, target_y, target_size, target_size])  # draws target
        pygame.draw.rect(window, black, [seeker_x, seeker_y, seeker_size, seeker_size])  # draws seeker

        # borders/maze walls {NB - ugly code, can definitely clean}
        pygame.draw.rect(window, black, [0, 0, 500, 20])  # outer border - top
        pygame.draw.rect(window, black, [0, 480, 500, 20])  # outer border - bottom
        pygame.draw.rect(window, black, [0, 0, 20, 500])  # outer border - left
        pygame.draw.rect(window, black, [480, 0, 20, 500])  # outer border - right
        pygame.draw.rect(window, black, [20, 120, 220, 20])  # wall - horizontal, top left
        pygame.draw.rect(window, black, [240, 120, 20, 260])  # wall - vertical, middle
        pygame.draw.rect(window, black, [120, 360, 120, 20])  # wall - horizontal, bottom middle
        pygame.draw.rect(window, black, [120, 240, 20, 120])  # wall - vertical, bottom left
        pygame.draw.rect(window, black, [360, 120, 20, 140])  # wall - vertical, top right
        pygame.draw.rect(window, black, [360, 240, 120, 20])  # wall - horizontal, top right
        pygame.draw.rect(window, black, [360, 360, 20, 120])  # wall - vertical, bottom right

        # score/move/timekeeping displays
        display_text("Score: " + str(score), white, 30, 5, 20)
        display_text('Timer: ' + str(round(seconds, 1)), white, 150, 5, 20)
        display_text("State: " + str(current_state), white, 270, 5, 20)
        display_text("Best action: " + str(optimal_action), white, 390, 5, 20)
        display_text("Moves: " + str(moves), white, 150, 485, 20)
        display_text("Moves this level: " + str(moves_this_level), white, 270, 485, 20)

        pygame.display.update()  # updates the screen

        for event in pygame.event.get():  # gets events from the queue; conditions below indicate event outcomes

            if event.type == pygame.QUIT:
                game_quit = True  # close window: quit game

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    game_quit = True  # press q: quit game
                elif event.key == pygame.K_i:
                    time_interval = 1000  # press i: set game speed to slowest (1 move per second)
                elif event.key == pygame.K_o:
                    time_interval = 200  # press o: set game speed to medium (5 moves per second)
                elif event.key == pygame.K_p:
                    time_interval = 100  # press p: set game speed to fastest (10 moves per second)

        # timekeeping
        ticker = clock.tick()
        time_since_last_update += ticker

        # conditional "implements action" determined by q-value table at given time intervals {NB: can be much cleaner}
        if time_since_last_update > time_interval:

            seconds += time_interval / 1000  # converts time interval value from ms to s
            moves += 1
            moves_this_level += 1

            # updates game conditions with values corresponding to each action {NB: same as timer, can be much cleaner}
            if optimal_action == 0:
                target_hit = True
            elif optimal_action == 1:
                seeker_y += - move_distance
                current_state = next_state
                optimal_action, next_state = optimal_route(current_state)
            elif optimal_action == 2:
                seeker_x += move_distance
                current_state = next_state
                optimal_action, next_state = optimal_route(current_state)
            elif optimal_action == 3:
                seeker_y += move_distance
                current_state = next_state
                optimal_action, next_state = optimal_route(current_state)
            elif optimal_action == 4:
                seeker_x += - move_distance
                current_state = next_state
                optimal_action, next_state = optimal_route(current_state)
            else:
                pass
            time_since_last_update = 0

        # updates score and sets new state when seeker at target coordinates and goal-achieving action performed
        if target_hit and seeker_x == target_x + 10 and seeker_y == target_y + 10:
            score += 1
            moves_this_level = 0
            target_hit = False
            current_state, seeker_x, seeker_y, target_x, target_y = xy_generator()
            optimal_action, next_state = optimal_route(current_state)

    pygame.quit()  # un-initialises and quits game when called (i.e. when while loop broken)
    quit()


pathfinder_game()  # opens and starts the game; program will finish when closed once
