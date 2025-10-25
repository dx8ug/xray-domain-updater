BIN_DIR := $(HOME)/.local/bin
CONFIG_DIR := $(HOME)/.config/xray-domain-updater
SCRIPT := xray-domain-updater.py

.PHONY: install uninstall

install:
	@mkdir -p $(BIN_DIR)
	@mkdir -p $(CONFIG_DIR)
	@chmod +x $(SCRIPT)
	@ln -sf $(CURDIR)/$(SCRIPT) $(BIN_DIR)/xray-domain-updater
	@cp -f config.json $(CONFIG_DIR)/config.json
	@chmod 600 $(CONFIG_DIR)/config.json

uninstall:
	@rm -f $(BIN_DIR)/xray-domain-updater
	@rm -rf $(CONFIG_DIR)

