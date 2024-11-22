import math
import random
import time
import matplotlib.pyplot as plt


class Nim:
    def __init__(self, initial=[1, 3, 5, 7]):
        self.piles = initial.copy()
        self.player = 0
        self.winner = None

    @classmethod
    def available_actions(cls, piles):
        actions = set()
        for i, pile in enumerate(piles):
            for j in range(1, pile + 1):
                actions.add((i, j))
        return actions

    @classmethod
    def other_player(cls, player):
        return 0 if player == 1 else 1

    def switch_player(self):
        self.player = Nim.other_player(self.player)

    def move(self, action):
        pile, count = action
        if self.winner is not None:
            raise Exception("Game already won")
        elif pile < 0 or pile >= len(self.piles):
            raise Exception("Invalid pile")
        elif count < 1 or count > self.piles[pile]:
            raise Exception("Invalid number of objects")

        self.piles[pile] -= count
        self.switch_player()

        if all(pile == 0 for pile in self.piles):
            self.winner = self.player


class NimAI:
    def __init__(self, alpha=0.5, epsilon=0.1, gamma=1):
        self.q = dict()
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma

    def update(self, old_state, action, new_state, reward):
        old = self.get_q_value(old_state, action)
        best_future = self.best_future_reward(new_state)
        self.update_q_value(old_state, action, old, reward, best_future)

    def get_q_value(self, state, action):
        return self.q.get((tuple(state), action), 0)

    def update_q_value(self, state, action, old_q, reward, future_rewards):
        new_value = reward + self.gamma * future_rewards
        self.q[(tuple(state), action)] = old_q + self.alpha * (new_value - old_q)

    def best_future_reward(self, state):
        actions = Nim.available_actions(state)
        if not actions:
            return 0
        return max(self.get_q_value(state, action) for action in actions)

    def choose_action(self, state, epsilon=True):
        actions = list(Nim.available_actions(state))
        if not actions:
            raise Exception("No available actions")

        if epsilon and random.random() < self.epsilon:
            return random.choice(actions)

        best_action = max(actions, key=lambda action: self.get_q_value(state, action))
        return best_action


def train(n):
    ai = NimAI()
    rewards = []

    for i in range(n):
        print(f"Playing training game {i + 1}")
        game = Nim()
        total_reward = 0
        last = {
            0: {"state": None, "action": None},
            1: {"state": None, "action": None}
        }

        while True:
            state = game.piles.copy()
            action = ai.choose_action(state)

            last[game.player]["state"] = state
            last[game.player]["action"] = action


            game.move(action)
            new_state = game.piles.copy()
            if game.winner is not None:
                # Player who made the last move loses
                ai.update(state, action, new_state, -1)

                # The other player wins
                ai.update(
                    last[game.player]["state"],
                    last[game.player]["action"],
                    new_state,
                    1
                )
                if game.winner == 1:
                    total_reward += 1  # Add reward to total (AI won)
                else:
                    total_reward -= 1  # Add reward to total (AI lost)

                rewards.append(total_reward)
                break

            # If game is continuing, no rewards yet
            elif last[game.player]["state"] is not None:
                ai.update(
                    last[game.player]["state"],
                    last[game.player]["action"],
                    new_state,
                    0
                )

    print("Done training")
    return ai, rewards


def plot_eval(rewards):
    cumulative_rewards = [sum(rewards[:i+1]) for i in range(len(rewards))]
    plt.figure(figsize=(10, 6))
    plt.plot(cumulative_rewards, label="Cumulative Rewards")
    plt.xlabel("Games")
    plt.ylabel("Cumulative Reward")
    plt.title("AI Training Performance")
    plt.legend()
    plt.show()


def play(ai, human_player=None):
    if human_player is None:
        human_player = random.randint(0, 1)

    game = Nim()
    while True:
        print("\nPiles:")
        for i, pile in enumerate(game.piles):
            print(f"Pile {i}: {pile}")

        available_actions = Nim.available_actions(game.piles)
        time.sleep(1)

        if game.player == human_player:
            print("Your Turn")
            while True:
                pile = int(input("Choose Pile: "))
                count = int(input("Choose Count: "))
                if (pile, count) in available_actions:
                    break
                print("Invalid move, try again.")
        else:
            print("AI's Turn")
            pile, count = ai.choose_action(game.piles, epsilon=False)
            print(f"AI chose to take {count} from pile {pile}.")

        game.move((pile, count))

        if game.winner is not None:
            print("\nGAME OVER")
            winner = "Human" if game.winner == human_player else "AI"
            print(f"Winner is {winner}")
            return


