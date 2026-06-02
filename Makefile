COMPOSE := docker compose

COMPOSE_FILE := docker-compose.yml

ARGS ?=

WS ?= $(CURDIR)
AGENT__WORKSPACE ?= $(WS)
export AGENT__WORKSPACE

up:
	$(COMPOSE) -f $(COMPOSE_FILE) up -d $(ARGS)
down:
	$(COMPOSE) -f $(COMPOSE_FILE) down $(ARGS)
down-v:
	$(COMPOSE) -f $(COMPOSE_FILE) down -v
build:
	$(COMPOSE) -f $(COMPOSE_FILE) build

restart: down up
restart-v: down-v up
rebuild: down build up
rebuild-v: down-v build up
