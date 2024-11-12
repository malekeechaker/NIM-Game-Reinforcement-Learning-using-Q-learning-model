import math
import random
import time
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

# Définition de l'Environnement de Jeu
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
            raise Exception("Partie déjà gagnée")
        elif pile < 0 or pile >= len(self.piles):
            raise Exception("Pile invalide")
        elif count < 1 or count > self.piles[pile]:
            raise Exception("Nombre d'objets invalide")
        self.piles[pile] -= count
        self.switch_player()
        if all(pile == 0 for pile in self.piles):
            self.winner = self.player

# Définition du Réseau de Neurones
class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Agent IA utilisant DQN
class NimAI:
    def __init__(self, state_size, action_size, alpha=0.001, gamma=0.9, epsilon=0.1, buffer_size=10000, batch_size=32, target_update=10):
        self.state_size = state_size
        self.action_size = action_size
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.target_update = target_update

        # Initialisation du modèle et du réseau cible
        self.model = QNetwork(state_size, action_size)
        self.target_model = QNetwork(state_size, action_size)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.alpha)

        # Replay buffer
        self.memory = deque(maxlen=buffer_size)
        self.steps = 0

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def choose_action(self, state, epsilon=True):
        if epsilon and random.random() < self.epsilon:
            return random.randrange(self.action_size)
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.model(state_tensor)
            return torch.argmax(q_values).item()

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze()
        next_q_values = self.target_model(next_states).max(1)[0]
        expected_q_values = rewards + (self.gamma * next_q_values * (1 - dones))

        loss = nn.MSELoss()(q_values, expected_q_values.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.steps += 1
        if self.steps % self.target_update == 0:
            self.target_model.load_state_dict(self.model.state_dict())

def train(n_games):
    ai = NimAI(state_size=4, action_size=16)  # Choisir des tailles appropriées pour Nim
    rewards = []

    for episode in range(n_games):
        game = Nim()
        total_reward = 0
        state = game.piles.copy()

        while game.winner is None:
            action = ai.choose_action(state)
            pile, count = action // 4, action % 4 + 1  # Convertir action en (pile, count)
            if (pile, count) not in Nim.available_actions(game.piles):
                reward = -1
                done = True
            else:
                game.move((pile, count))
                next_state = game.piles.copy()
                reward = 1 if game.winner == ai.player else 0
                done = game.winner is not None
                ai.remember(state, action, reward, next_state, done)
                total_reward += reward
                state = next_state
            ai.replay()
        rewards.append(total_reward)

    return ai, rewards

def plot_eval(rewards):
    plt.figure(figsize=(10, 6))
    plt.plot(rewards, label="Récompense cumulée")
    plt.xlabel("Jeux joués")
    plt.ylabel("Récompense cumulée")
    plt.title("Récompense cumulée au cours de l'entraînement")
    plt.legend()
    plt.show()

def play(ai):
    game = Nim()
    while game.winner is None:
        print(f"\nPiles: {game.piles}")
        if game.player == 0:
            pile = int(input("Choisissez une pile: "))
            count = int(input("Choisissez combien retirer: "))
        else:
            state = game.piles.copy()
            action = ai.choose_action(state, epsilon=False)
            pile, count = action // 4, action % 4 + 1
            print(f"L'IA retire {count} de la pile {pile}")
        game.move((pile, count))
    print("Fin de la partie!")
    print("Gagnant:", "Humain" if game.winner == 0 else "IA")
