import sys
import math
import pygame


def build_reward_gates(num_gates, center, radius, gate_w, gate_h):
	"""Return a list of gate dicts (center, w, h, angle) placed evenly around a circle."""
	gates = []
	cx, cy = center
	for i in range(num_gates):
		ang = i * (360.0 / num_gates)
		rad = math.radians(ang)
		gx = cx + math.cos(rad) * radius
		gy = cy + math.sin(rad) * radius
		gate_angle = (ang + 90) % 360
		gates.append({"center": (gx, gy), "w": gate_w, "h": gate_h, "angle": gate_angle})
	return gates


def point_in_rotated_rect(px, py, rect_center, w, h, angle_deg):
	cx, cy = rect_center
	dx = px - cx
	dy = py - cy
	rad = math.radians(-angle_deg)
	cos_a = math.cos(rad)
	sin_a = math.sin(rad)
	lx = dx * cos_a - dy * sin_a
	ly = dx * sin_a + dy * cos_a
	return (abs(lx) <= w / 2) and (abs(ly) <= h / 2)


def draw_gate(surface, gate, color=(0, 200, 0, 180)):
	# Create a transparent surface for the gate rectangle, rotate and blit centered
	w, h = int(gate["w"]), int(gate["h"])
	surf = pygame.Surface((w, h), pygame.SRCALPHA)
	surf.fill(color)
	rot = pygame.transform.rotate(surf, gate["angle"])
	r = rot.get_rect(center=(int(gate["center"][0]), int(gate["center"][1])))
	surface.blit(rot, r.topleft)


def main():
	pygame.init()
	screen_w, screen_h = 800, 600
	screen = pygame.display.set_mode((screen_w, screen_h))
	pygame.display.set_caption("reward gates test")
	clock = pygame.time.Clock()

	# Track parameters (ellipse)
	bg = (30, 30, 30)
	track_color = (200, 200, 200)
	track_width = 20
	margin = track_width // 2 + 5
	track_rect = pygame.Rect(margin, margin, screen_w - 2 * margin, screen_h - 2 * margin)

	# Gates: build a few around the circle
	center = (screen_w / 2, screen_h / 2)
	gates = build_reward_gates(num_gates=8, center=center, radius=180, gate_w=8, gate_h=48)
	current_gate = 0
	total_reward = 0.0

	# Simple car state (circle) controlled by keys
	car_x, car_y = center[0], center[1] + 180  # start near bottom of circle
	car_speed = 0.0
	car_angle = -90  # facing right-ish; we'll allow simple movement

	font = pygame.font.SysFont(None, 24)

	running = True
	while running:
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				running = False

		keys = pygame.key.get_pressed()
		# Basic control: arrow keys to move the car point around
		if keys[pygame.K_UP] or keys[pygame.K_w]:
			car_speed += 0.3
		if keys[pygame.K_DOWN] or keys[pygame.K_s]:
			car_speed -= 0.3
		if keys[pygame.K_LEFT] or keys[pygame.K_a]:
			car_angle += 4
		if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
			car_angle -= 4

		# friction and clamp
		car_speed *= 0.95
		car_speed = max(min(car_speed, 8), -4)

		# move
		rad = math.radians(-car_angle)
		car_x += math.cos(rad) * car_speed
		car_y += math.sin(rad) * car_speed

		# Keep inside window
		car_x = max(0, min(screen_w - 1, car_x))
		car_y = max(0, min(screen_h - 1, car_y))

		# Draw scene
		screen.fill(bg)
		pygame.draw.ellipse(screen, track_color, track_rect, width=track_width)

		# Draw gates
		for i, g in enumerate(gates):
			color = (0, 200, 0, 160) if i == current_gate else (100, 100, 100, 120)
			draw_gate(screen, g, color=color)

		# Draw car as a small rotated rectangle for visibility
		car_w, car_h = 20, 12
		car_surf = pygame.Surface((car_w, car_h), pygame.SRCALPHA)
		car_surf.fill((255, 50, 50))
		car_rot = pygame.transform.rotate(car_surf, car_angle)
		car_r = car_rot.get_rect(center=(int(car_x), int(car_y)))
		screen.blit(car_rot, car_r.topleft)

		# Check gate collision using car center point in rotated rect
		hit = point_in_rotated_rect(car_x, car_y, gates[current_gate]["center"], gates[current_gate]["w"], gates[current_gate]["h"], gates[current_gate]["angle"])
		if hit:
			total_reward += 10.0
			current_gate = (current_gate + 1) % len(gates)

		# HUD
		score_text = font.render(f"Reward: {total_reward:.1f}", True, (255, 255, 255))
		gate_text = font.render(f"Next gate: {current_gate}", True, (255, 255, 255))
		screen.blit(score_text, (10, 10))
		screen.blit(gate_text, (10, 40))

		pygame.display.flip()
		clock.tick(60)

	pygame.quit()


if __name__ == "__main__":
	try:
		main()
	except Exception as e:
		print("Error while running pygamedraw_test:", e)
		pygame.quit()
		sys.exit(1)