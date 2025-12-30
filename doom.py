import os
from random import choice
from time import sleep

import numpy as np
import vizdoom as vzd
from PIL import Image


def init_doom(game: vzd.DoomGame):
    # Now it's time for configuration!
    # load_config could be used to load configuration instead of doing it here with code.
    # If load_config is used in-code configuration will also work - most recent changes will add to previous ones.
    game.load_config("./config/doom-config.cfg")
    game.init()


def save_screen_buffer(screen_buf, frame_number: int):
    # Transpose from (channels, height, width) to (height, width, channels) for PIL
    screen_buf = np.transpose(screen_buf, (1, 2, 0))
    img = Image.fromarray(screen_buf)
    if not os.path.exists("screens"):
        os.makedirs("screens")
    img.save(f"screens/screen_{frame_number:04d}.png")

def main():
    pass

if __name__ == "__main__":
    # Create DoomGame instance. It will run the game and communicate with you.
    game = vzd.DoomGame()
    init_doom(game)

    actions = [[True, False, False], [
        False, True, False], [False, False, True]]
    
    # Run this many episodes
    episodes = 10

    # Sets time that will pause the engine after each action (in seconds)
    # Without this everything would go too fast for you to keep track of what's happening.
    sleep_time = 1.0

    for i in range(episodes):
        print(f"Episode #{i + 1}")

        # Starts a new episode. It is not needed right after init() but it doesn't cost much. At least the loop is nicer.
        game.new_episode()

        while not game.is_episode_finished():

            # Gets the state
            state = game.get_state()
            assert state is not None

            # Which consists of:
            n = state.number
            vars = state.game_variables

            # Different buffers (screens, depth, labels, automap, audio)
            # Expect of screen buffer some may be None if not first enabled.
            screen_buf = state.screen_buffer
            depth_buf = state.depth_buffer
            labels_buf = state.labels_buffer
            automap_buf = state.automap_buffer
            audio_buf = state.audio_buffer

            print("=" * 80)
            print("*")
            print("* Screen buffer size:", None if screen_buf is None else screen_buf.shape)
            print("* Depth buffer size:", None if depth_buf is None else depth_buf.shape)
            print("* Game var (Health):", vars[0])
            print("* Game var (Ammo2):", vars[1])
            print("* Game var (Position X):", vars[2])
            print("* Game var (Position Y):", vars[3])
            print("* Game var (Velocity X):", vars[4])
            print("* Game var (Velocity Y):", vars[5])
            print("=" * 80)

            # Save the screen buffer to an image file
            if screen_buf is not None:
                save_screen_buffer(screen_buf, n)

            # List of labeled objects visible in the frame, may be None if not first enabled.
            labels = state.labels

            # List of all objects (enemies, pickups, etc.) present in the current episode, may be None if not first enabled
            objects = state.objects

            # List of all sectors (map geometry), may be None if not first enabled.
            sectors = state.sectors

            # Games variables can be also accessed via
            # (including the ones that were not added as available to a game state):
            # game.get_game_variable(GameVariable.AMMO2)

            # Makes an action (here random one) and returns a reward.
            r = game.make_action(choice(actions))

            # Makes a "prolonged" action and skip frames:
            # skiprate = 4
            # r = game.make_action(choice(actions), skiprate)

            # The same could be achieved with:
            # game.set_action(choice(actions))
            # game.advance_action(skiprate)
            # r = game.get_last_reward()

            # Prints state's game variables and reward.
            print(f"State #{n}")
            print("Game variables:", vars)
            print("Reward:", r)
            print("=====================")

            if sleep_time > 0:
                sleep(sleep_time)

        # Check how the episode went.
        print("Episode finished.")
        print("Total reward:", game.get_total_reward())
        print("************************")

    # It will be done automatically anyway but sometimes you need to do it in the middle of the program...
    game.close()
