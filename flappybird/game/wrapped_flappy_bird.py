import numpy as np
import sys
import random
import pygame
import flappy_bird_utils
from itertools import cycle

# FPS = 3000
# SCREENWIDTH = 288
# SCREENHEIGHT = 512

# pygame.init()
# FPSCLOCK = pygame.time.Clock()
# SCREEN = pygame.display.set_mode((SCREENWIDTH, SCREENHEIGHT))
# pygame.display.set_caption('Flappy Bird')
# #
# IMAGES, SOUNDS, HITMASKS = flappy_bird_utils.load()
# PIPEGAPSIZE = 100 # gap between upper and lower part of pipe
# BASEY = SCREENHEIGHT * 0.79
#
# PLAYER_WIDTH = IMAGES['player'][0].get_width()
# PLAYER_HEIGHT = IMAGES['player'][0].get_height()
# PIPE_WIDTH = IMAGES['pipe'][0].get_width()
# PIPE_HEIGHT = IMAGES['pipe'][0].get_height()
# BACKGROUND_WIDTH = IMAGES['background'].get_width()
#
# PLAYER_INDEX_GEN = cycle([0, 1, 2, 1])

class GameState:
    def __init__(self,cfg):
        pygame.init()
        self.FPS = cfg.FPS
        self.SCREENWIDTH = cfg.SCREENWIDTH
        self.SCREENHEIGHT = cfg.SCREENHEIGHT

        self.SCREEN = pygame.display.set_mode((self.SCREENWIDTH, self.SCREENHEIGHT))
        self.FPSCLOCK = pygame.time.Clock()
        pygame.display.set_caption('Flappy Bird')
        self.IMAGES, SOUNDS, self.HITMASKS = flappy_bird_utils.load()
        self.PLAYER_WIDTH = self.IMAGES['player'][0].get_width()
        self.PLAYER_HEIGHT = self.IMAGES['player'][0].get_height()
        self.PIPE_WIDTH = self.IMAGES['pipe'][0].get_width()
        self.PIPE_HEIGHT = self.IMAGES['pipe'][0].get_height()
        self.BACKGROUND_WIDTH = self.IMAGES['background'].get_width()
        self.PLAYER_INDEX_GEN = cycle([0, 1, 2, 1])
        self.last_score = 0
        self.score = self.playerIndex = self.loopIter = 0
        self.playerx = int(self.SCREENWIDTH * 0.2)
        self.playery = int((self.SCREENHEIGHT - self.PLAYER_HEIGHT) / 2)
        self.basex = 0
        self.baseShift = self.IMAGES['base'].get_width() - self.BACKGROUND_WIDTH
        self.PIPEGAPSIZE = 100  # gap between upper and lower part of pipe
        self.BASEY = self.SCREENHEIGHT * 0.79
        newPipe1 = self.getRandomPipe()
        newPipe2 = self.getRandomPipe()
        self.upperPipes = [
            {'x': self.SCREENWIDTH, 'y': newPipe1[0]['y']},
            {'x': self.SCREENWIDTH + (self.SCREENWIDTH / 2), 'y': newPipe2[0]['y']},
        ]
        self.lowerPipes = [
            {'x': self.SCREENWIDTH, 'y': newPipe1[1]['y']},
            {'x': self.SCREENWIDTH + (self.SCREENWIDTH / 2), 'y': newPipe2[1]['y']},
        ]
        # player velocity, max velocity, downward accleration, accleration on flap
        self.pipeVelX = -4
        self.playerVelY = 0  # player's velocity along Y, default same as playerFlapped
        self.playerMaxVelY = 10  # max vel along Y, max descend speed
        self.playerMinVelY = -8  # min vel along Y, max ascend speed
        self.playerAccY = 1  # players downward accleration
        self.playerFlapAcc = -9  # players speed on flapping
        self.playerFlapped = False  # True when player flaps

    def reset(self, last_score):
        self.last_score = last_score
        self.score = self.playerIndex = self.loopIter = 0
        self.playerx = int(self.SCREENWIDTH * 0.2)
        self.playery = int((self.SCREENHEIGHT - self.PLAYER_HEIGHT) / 2)
        self.basex = 0
        self.baseShift = self.IMAGES['base'].get_width() - self.BACKGROUND_WIDTH
        self.PIPEGAPSIZE = 100  # gap between upper and lower part of pipe
        self.BASEY = self.SCREENHEIGHT * 0.79
        newPipe1 = self.getRandomPipe()
        newPipe2 = self.getRandomPipe()
        self.upperPipes = [
            {'x': self.SCREENWIDTH, 'y': newPipe1[0]['y']},
            {'x': self.SCREENWIDTH + (self.SCREENWIDTH / 2), 'y': newPipe2[0]['y']},
        ]
        self.lowerPipes = [
            {'x': self.SCREENWIDTH, 'y': newPipe1[1]['y']},
            {'x':self. SCREENWIDTH + (self.SCREENWIDTH / 2), 'y': newPipe2[1]['y']},
        ]
        # player velocity, max velocity, downward accleration, accleration on flap
        self.pipeVelX = -4
        self.playerVelY    =  0    # player's velocity along Y, default same as playerFlapped
        self.playerMaxVelY =  10   # max vel along Y, max descend speed
        self.playerMinVelY =  -8   # min vel along Y, max ascend speed
        self.playerAccY    =   1   # players downward accleration
        self.playerFlapAcc =  -9   # players speed on flapping
        self.playerFlapped = False # True when player flaps

    def frame_step(self, input_actions):
        pygame.event.pump()
        reward = 0.1
        terminal = False

        if sum(input_actions) != 1:
            raise ValueError('Multiple input actions!')

        # input_actions[0] == 1: do nothing
        # input_actions[1] == 1: flap the bird
        if input_actions[1] == 1:
            if self.playery > -2 * self.PLAYER_HEIGHT:
                self.playerVelY = self.playerFlapAcc
                self.playerFlapped = True
                #SOUNDS['wing'].play()

        # check for score
        playerMidPos = self.playerx + self.PLAYER_WIDTH / 2
        for pipe in self.upperPipes:
            pipeMidPos = pipe['x'] + self.PIPE_WIDTH / 2
            if pipeMidPos <= playerMidPos < pipeMidPos + 4:
                self.score += 1
                #SOUNDS['point'].play()
                reward = 1

        # playerIndex basex change
        if (self.loopIter + 1) % 3 == 0:
            self.playerIndex = next(self.PLAYER_INDEX_GEN)
        self.loopIter = (self.loopIter + 1) % 30
        self.basex = -((-self.basex + 100) % self.baseShift)

        # player's movement
        if self.playerVelY < self.playerMaxVelY and not self.playerFlapped:
            self.playerVelY += self.playerAccY
        if self.playerFlapped:
            self.playerFlapped = False

        self.playery += self.playerVelY
        if self.playery > self.BASEY - self.playery - self.PLAYER_HEIGHT :
            self.playery = self.BASEY - self.playery - self.PLAYER_HEIGHT
            reward -=0.1
        # self.playery += min(self.playerVelY, BASEY - self.playery - PLAYER_HEIGHT)
        if self.playery < 0:
            self.playery = 0
            reward -= 0.1

        # move pipes to left
        for uPipe, lPipe in zip(self.upperPipes, self.lowerPipes):
            uPipe['x'] += self.pipeVelX
            lPipe['x'] += self.pipeVelX

        # add new pipe when first pipe is about to touch left of screen
        if 0 < self.upperPipes[0]['x'] < 5:
            newPipe = self.getRandomPipe()
            self.upperPipes.append(newPipe[0])
            self.lowerPipes.append(newPipe[1])

        # remove first pipe if its out of the screen
        if self.upperPipes[0]['x'] < -self.PIPE_WIDTH:
            self.upperPipes.pop(0)
            self.lowerPipes.pop(0)

        # check if crash here
        isCrash= self.checkCrash({'x': self.playerx, 'y': self.playery,
                             'index': self.playerIndex},
                            self.upperPipes, self.lowerPipes)
        if isCrash:
            #SOUNDS['hit'].play()
            #SOUNDS['die'].play()
            terminal = True
            # print("game end,sorce is :", self.get_score())
            self.reset(self.get_score())
            reward = -1

        # draw sprites
        self.SCREEN.blit(self.IMAGES['background'], (0,0))

        for uPipe, lPipe in zip(self.upperPipes, self.lowerPipes):
            self.SCREEN.blit(self.IMAGES['pipe'][0], (uPipe['x'], uPipe['y']))
            self.SCREEN.blit(self.IMAGES['pipe'][1], (lPipe['x'], lPipe['y']))

        self.SCREEN.blit(self.IMAGES['base'], (self.basex, self.BASEY))
        # print score so player overlaps the score
        self.showScore(self.score)
        self.SCREEN.blit(self.IMAGES['player'][self.playerIndex],
                    (self.playerx, self.playery))

        image_data = pygame.surfarray.array3d(pygame.display.get_surface())
        pygame.display.update()
        self.FPSCLOCK.tick(self.FPS)
        #print self.upperPipes[0]['y'] + PIPE_HEIGHT - int(BASEY * 0.2)
        return image_data, reward, terminal

    def frame_step_v2(self, input_actions):
        pygame.event.pump()

        reward_t = 0
        imagedata_t = 0
        terminal_t = False
        a_t = np.zeros([2]).astype(int)

        if sum(input_actions) != 1:
            raise ValueError('Multiple input actions!')

        # input_actions[0] == 1: do nothing
        # input_actions[1] == 1: flap the bird
        # input_actions[2] == 1: flap the bird twice
        # input_actions[3] == 1: flap the bird third times
        if input_actions[0] == 1:
            a_t[0] = 1
            return self.frame_step(a_t)

        if input_actions[1] == 1:
            a_t[1] = 1
            imagedata_t, reward_t, terminal_t = self.frame_step(a_t)
            return imagedata_t, reward_t-0.05, terminal_t

        if input_actions[2] == 1:
            a_t[1] = 1
            for i in range(2):
                imagedata_t, reward_t, terminal_t = self.frame_step(a_t)
                pygame.event.pump()
                # reward += reward_t
                if terminal_t:
                    return imagedata_t, reward_t - 0.1*i, terminal_t
            return imagedata_t, reward_t - 0.2, terminal_t

        if input_actions[3] == 1:
            a_t[1] = 1
            for i in range(3):
                imagedata_t, reward_t, terminal_t = self.frame_step(a_t)
                pygame.event.pump()
                # reward += reward_t
                if terminal_t:
                    return imagedata_t, reward_t - 0.1*i, terminal_t
            return imagedata_t, reward_t - 0.3, terminal_t

    def get_score(self):
        return self.score

    def get_last_score(self):
        return self.last_score

    def getRandomPipe(self):
        """returns a randomly generated pipe"""
        # y of gap between upper and lower pipe
        gapYs = [20, 30, 40, 50, 60, 70, 80, 90]
        index = random.randint(0, len(gapYs)-1)
        gapY = gapYs[index]

        gapY += int(self.BASEY * 0.2)
        pipeX = self.SCREENWIDTH + 10

        return [
            {'x': pipeX, 'y': gapY - self.PIPE_HEIGHT},  # upper pipe
            {'x': pipeX, 'y': gapY + self.PIPEGAPSIZE},  # lower pipe
        ]

    def showScore(self, score):
        """displays score in center of screen"""
        scoreDigits = [int(x) for x in list(str(score))]
        totalWidth = 0 # total width of all numbers to be printed

        for digit in scoreDigits:
            totalWidth += self.IMAGES['numbers'][digit].get_width()

        Xoffset = (self.SCREENWIDTH - totalWidth) / 2

        for digit in scoreDigits:
            self.SCREEN.blit(self.IMAGES['numbers'][digit], (Xoffset, self.SCREENHEIGHT * 0.1))
            Xoffset += self.IMAGES['numbers'][digit].get_width()

    def checkCrash(self, player, upperPipes, lowerPipes):
        """returns True if player collders with base or pipes."""
        pi = player['index']
        player['w'] = self.IMAGES['player'][0].get_width()
        player['h'] = self.IMAGES['player'][0].get_height()

        # if player crashes into ground
        if player['y'] + player['h'] >= self.BASEY - 1:
            return True
        else:

            playerRect = pygame.Rect(player['x'], player['y'],
                          player['w'], player['h'])

            for uPipe, lPipe in zip(upperPipes, lowerPipes):
                # upper and lower pipe rects
                uPipeRect = pygame.Rect(uPipe['x'], uPipe['y'], self.PIPE_WIDTH, self.PIPE_HEIGHT)
                lPipeRect = pygame.Rect(lPipe['x'], lPipe['y'], self.PIPE_WIDTH, self.PIPE_HEIGHT)

                # player and upper/lower pipe hitmasks
                pHitMask = self.HITMASKS['player'][pi]
                uHitmask = self.HITMASKS['pipe'][0]
                lHitmask = self.HITMASKS['pipe'][1]

                # if bird collided with upipe or lpipe
                uCollide = pixelCollision(playerRect, uPipeRect, pHitMask, uHitmask)
                lCollide = pixelCollision(playerRect, lPipeRect, pHitMask, lHitmask)

                if uCollide or lCollide:
                    return True

        return False


def pixelCollision(rect1, rect2, hitmask1, hitmask2):
    """Checks if two objects collide and not just their rects"""
    rect = rect1.clip(rect2)

    if rect.width == 0 or rect.height == 0:
        return False

    x1, y1 = rect.x - rect1.x, rect.y - rect1.y
    x2, y2 = rect.x - rect2.x, rect.y - rect2.y

    for x in range(rect.width):
        for y in range(rect.height):
            if hitmask1[x1+x][y1+y] and hitmask2[x2+x][y2+y]:
                return True
    return False
