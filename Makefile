COMPOSE := docker compose --env-file .langfuse.env

COMPOSE_FILE := docker-compose.yml
COMPOSE_FILE_OLLAMA := docker-compose-ollama.yml

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

up-o:
	$(COMPOSE) -f $(COMPOSE_FILE_OLLAMA) up -d $(ARGS)
down-o:
	$(COMPOSE) -f $(COMPOSE_FILE_OLLAMA) down $(ARGS)
build-o:
	$(COMPOSE) -f $(COMPOSE_FILE_OLLAMA) build
