#!/usr/bin/env python3

import sys
import random
from pathlib import Path
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget
from PyQt6.QtGui import QPainter, QPixmap, QKeyEvent, QFont, QColor
from PyQt6.QtCore import Qt, QTimer, QRect, QUrl
from PyQt6.QtMultimedia import QSoundEffect

BRIGHT_COLORS = [
    QColor(255, 0, 0), QColor(0, 255, 0), QColor(0, 0, 255),
    QColor(255, 255, 0), QColor(255, 0, 255), QColor(0, 255, 255)
]

class Alien:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.alive = True
        self.color = random.choice(BRIGHT_COLORS)

class SpaceInvadersGame(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(1000, 800)
        self.player_width = 60
        self.player_height = 40

        # Game state
        self.reset_game(full_reset=True)

        # Sprites
        self.player_sprite = QPixmap("player.png").scaled(60, 40)
        self.bullet_sprite = QPixmap("bullet.png").scaled(5, 15)
        self.alien_bullet_sprite = QPixmap("alien_bullet.png").scaled(5, 15)

        # Sounds
        self.shoot_sound = QSoundEffect()
        self.shoot_sound.setSource(QUrl.fromLocalFile(str(Path("shoot.wav").resolve())))
        self.shoot_sound.setVolume(0.5)

        self.hit_sound = QSoundEffect()
        self.hit_sound.setSource(QUrl.fromLocalFile(str(Path("hit.wav").resolve())))
        self.hit_sound.setVolume(0.5)

        self.alien_shoot_sound = QSoundEffect()
        self.alien_shoot_sound.setSource(QUrl.fromLocalFile(str(Path("alien_shoot.wav").resolve())))
        self.alien_shoot_sound.setVolume(0.5)

        # Timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_game)
        self.timer.start(50)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def reset_game(self, full_reset=False):
        self.player_x = 270
        self.player_y = 750
        self.bullets = []
        self.alien_bullets = []
        self.aliens = []
        self.alien_direction = 1
        self.alien_speed = 10
        self.score = 0 if full_reset else self.score
        self.lives = 3
        self.level = 1 if full_reset else self.level
        self.game_over = False
        self.started = False
        self.alien_shoot_cooldown = 0
        self.create_aliens()

    def create_aliens(self):
        self.aliens.clear()
        for i in range(5):
            for j in range(8):
                alien = Alien(50 + j*60, 50 + i*50)
                self.aliens.append(alien)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(0, 0, self.width(), self.height(), Qt.GlobalColor.black)

        if not self.started:
            painter.setPen(Qt.GlobalColor.white)
            painter.setFont(QFont("Arial", 32))
            painter.drawText(self.width()//2 - 150, self.height()//2, "PRESS S TO START")
            return

        painter.drawPixmap(self.player_x, self.player_y, self.player_sprite)

        for bx, by in self.bullets:
            painter.drawPixmap(bx, by, self.bullet_sprite)

        for alien in self.aliens:
            if alien.alive:
                painter.fillRect(alien.x, alien.y, 40, 30, alien.color)

        for bx, by in self.alien_bullets:
            painter.drawPixmap(bx, by, self.alien_bullet_sprite)

        painter.setPen(Qt.GlobalColor.white)
        painter.setFont(QFont("Arial", 16))
        painter.drawText(10, 20, f"Score: {self.score}  Lives: {self.lives}  Level: {self.level}")

        if self.game_over:
            painter.setFont(QFont("Arial", 36))
            painter.drawText(self.width()//2 - 120, self.height()//2, "GAME OVER")
            painter.setFont(QFont("Arial", 20))
            painter.drawText(self.width()//2 - 80, self.height()//2 + 40, "Press R to Restart")

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key.Key_Q:
            QApplication.quit()
            return

        if event.key() == Qt.Key.Key_S and not self.started:
            self.started = True
            return

        if not self.started:
            return

        if self.game_over:
            if event.key() == Qt.Key.Key_R:
                self.reset_game(full_reset=True)
            return

        if event.key() == Qt.Key.Key_Left and self.player_x > 0:
            self.player_x -= 20
        elif event.key() == Qt.Key.Key_Right and self.player_x < self.width() - self.player_width:
            self.player_x += 20
        elif event.key() == Qt.Key.Key_Space:
            self.bullets.append([self.player_x + self.player_width//2 - 2, self.player_y])
            self.shoot_sound.play()

    def update_game(self):
        if not self.started or self.game_over:
            self.update()
            return

        # Move bullets
        self.bullets = [[bx, by-20] for bx, by in self.bullets if by-20 > 0]
        self.alien_bullets = [[bx, by+15] for bx, by in self.alien_bullets if by+15 < self.height()]

        # Move aliens
        edge_hit = False
        for alien in self.aliens:
            alien.x += self.alien_direction * self.alien_speed
            if alien.x <= 0 or alien.x >= self.width() - 40:
                edge_hit = True
        if edge_hit:
            self.alien_direction *= -1
            for alien in self.aliens:
                alien.y += 20

        # Alien shooting
        if self.aliens and self.alien_shoot_cooldown <= 0:
            if random.random() < 0.3:
                shooter = random.choice([a for a in self.aliens if a.alive])
                self.alien_bullets.append([shooter.x + 18, shooter.y + 30])
                self.alien_shoot_sound.play()
                self.alien_shoot_cooldown = 20
        else:
            self.alien_shoot_cooldown -= 1

        # Player bullets hitting aliens
        for b in self.bullets[:]:
            bx, by = b
            for alien in self.aliens:
                if alien.alive and QRect(bx, by, 5, 15).intersects(QRect(alien.x, alien.y, 40, 30)):
                    alien.alive = False
                    self.score += 10
                    self.bullets.remove(b)
                    self.hit_sound.play()
                    break

        # Alien bullets hitting player
        for b in self.alien_bullets[:]:
            bx, by = b
            if QRect(bx, by, 5, 15).intersects(QRect(self.player_x, self.player_y, self.player_width, self.player_height)):
                self.lives -= 1
                self.alien_bullets.remove(b)
                if self.lives <= 0:
                    self.game_over = True

        # Level complete
        if all(not a.alive for a in self.aliens):
            self.level += 1
            self.alien_speed += 2
            self.create_aliens()

        # Game over if aliens reach player
        for alien in self.aliens:
            if alien.alive and alien.y + 30 >= self.player_y:
                self.game_over = True

        self.update()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = QMainWindow()
    window.setWindowTitle("Space Invaders")
    game = SpaceInvadersGame(window)
    window.setCentralWidget(game)
    window.show()
    sys.exit(app.exec())
