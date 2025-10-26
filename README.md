# Xray Domain Updater

A command-line utility for managing domain lists in Xray routing configuration on a remote server via SSH.

This tool simplifies the management of domain-based routing rules in Xray (VPN/proxy server) by providing easy-to-use commands for adding, removing, and listing domains, with automatic backup creation and service restart functionality.

## Features

- Add and remove domains from Xray routing rules
- List all configured domains with numbering
- Automatic backup creation before any configuration changes
- Safe configuration validation before applying changes
- Automatic service restart after modifications
- Backup management (list, restore, cleanup)
- Service status checking
- SSH-based remote configuration management
- Cascading configuration file search (script directory or XDG config directory)

## Requirements

- Python 3.10 or higher
- SSH access to remote server with Xray installed
- Private SSH key for authentication
- Remote server with xkeen service manager

## Installation

### Method 1: Using Makefile (Recommended)

```bash
make install
```

This will:
- Create symlink in `~/.local/bin/xray-domain-updater`
- Create config directory at `~/.config/xray-domain-updater/`
- Copy configuration file with secure permissions (600)

Make sure `~/.local/bin` is in your PATH.

### Method 2: Manual Installation

```bash
# Make script executable
chmod +x xray-domain-updater.py

# Create config directory
mkdir -p ~/.config/xray-domain-updater

# Copy and edit configuration
cp config.example.json ~/.config/xray-domain-updater/config.json
chmod 600 ~/.config/xray-domain-updater/config.json

# Edit configuration with your settings
nano ~/.config/xray-domain-updater/config.json

# Create symlink (optional)
ln -s $(pwd)/xray-domain-updater.py ~/.local/bin/xray-domain-updater
```

## Configuration

The utility uses a JSON configuration file that can be located in two places (checked in order):

1. `./config.json` (next to the script)
2. `~/.config/xray-domain-updater/config.json` (XDG config directory)

### Configuration File Structure

```json
{
    "ssh": {
        "host": "your.server.ip",
        "user": "root",
        "key_path": "~/.ssh/your_key"
    },
    "xray": {
        "remote_config_path": "/opt/etc/xray/configs/05_routing.json",
        "target_outbound_tag": "vless-reality"
    },
    "backup": {
        "dir": "/opt/etc/xray/backups",
        "count": 3
    }
}
```

See `config.example.json` for a template.

**Important**: Configuration file should have restrictive permissions (600 or 400) to protect sensitive data.

## Usage

### Basic Commands

#### List Domains

Display all configured domains with their index numbers:

```bash
xray-domain-updater list
```

Example output:
```
1. example.com
2. domain.net
3. site.org

Total domains: 3
```

#### Add Domain

Add a new domain to the routing configuration:

```bash
xray-domain-updater add example.com
```

The utility will:
1. Create a backup of current configuration
2. Add the domain to the list
3. Upload updated configuration to server
4. Restart the xkeen service
5. Verify service status

#### Remove Domain

Remove a domain by its index number from the list:

```bash
# Preview removal (shows which domain will be removed)
xray-domain-updater remove 2

# Confirm removal with -y flag
xray-domain-updater remove 2 -y
```

The utility will:
1. Create a backup of current configuration
2. Remove the domain from the list
3. Upload updated configuration to server
4. Restart the xkeen service
5. Verify service status

#### Restart Service

Manually restart the xkeen service:

```bash
xray-domain-updater restart
```

#### Check Service Status

Verify that xkeen service is running:

```bash
xray-domain-updater status
```

### Backup Management

#### List Backups

Show all available backups sorted by date (newest first):

```bash
xray-domain-updater backup list
```

Example output:
```
1. routing_20250126_143022.json (2025-01-26 14:30:22)
2. routing_20250126_120530.json (2025-01-26 12:05:30)
3. routing_20250125_183045.json (2025-01-25 18:30:45)

Total backups: 3
```

#### Restore from Backup

Preview backup content before restoring:

```bash
xray-domain-updater backup restore 1 --preview
```

Restore configuration from backup:

```bash
xray-domain-updater backup restore 1 --confirm
```

The utility will:
1. Restore the selected backup to the active configuration file
2. Restart the xkeen service
3. Verify service status

#### Clean Old Backups

Remove old backups exceeding the configured limit:

```bash
xray-domain-updater backup clean
```

Note: This is done automatically after each configuration change.

### Additional Options

#### Custom Configuration File

Specify an alternative configuration file path:

```bash
xray-domain-updater --config /path/to/config.json list
```

## Common Workflows

### Adding a New Domain

```bash
# Check current domains
xray-domain-updater list

# Add new domain
xray-domain-updater add newdomain.com

# Verify it was added
xray-domain-updater list
```

### Removing an Unwanted Domain

```bash
# List domains to find the index
xray-domain-updater list

# Preview removal
xray-domain-updater remove 3

# Confirm removal
xray-domain-updater remove 3 -y

# Verify it was removed
xray-domain-updater list
```

### Recovering from Configuration Error

If service fails after a change:

```bash
# Check service status
xray-domain-updater status

# List available backups
xray-domain-updater backup list

# Preview backup content
xray-domain-updater backup restore 1 --preview

# Restore the backup
xray-domain-updater backup restore 1 --confirm

# Verify service is working
xray-domain-updater status
```

## Configuration File Details

### SSH Section

- **host**: IP address or hostname of the remote server
- **user**: SSH username (typically 'root')
- **key_path**: Path to private SSH key file (supports tilde expansion)

### Xray Section

- **remote_config_path**: Full path to Xray routing configuration file on remote server
- **target_outbound_tag**: The outbound tag in routing rules where domains should be managed

### Backup Section

- **dir**: Directory on remote server where backups will be stored
- **count**: Maximum number of backups to keep (older backups are automatically deleted)

## Security Notes

### File Permissions

The utility checks and warns about insecure file permissions. Ensure proper permissions:

```bash
# Configuration file
chmod 600 ~/.config/xray-domain-updater/config.json

# SSH private key
chmod 600 ~/.ssh/your_key
```

### SSH Key Security

- Use a dedicated SSH key for this utility
- Never commit private keys to version control
- Consider using ssh-agent for key management
- Restrict SSH key permissions on the server side

### Configuration File

- Store configuration file in a secure location
- Do not share configuration files containing server credentials
- Use `.gitignore` to exclude config.json from version control

## Troubleshooting

### Configuration File Not Found

**Error**: `Configuration file not found`

**Solution**: Create configuration file in one of the expected locations:
```bash
# Option 1: Next to script
cp config.example.json config.json

# Option 2: XDG config directory
mkdir -p ~/.config/xray-domain-updater
cp config.example.json ~/.config/xray-domain-updater/config.json
```

Then edit the file with your actual server details.

### SSH Connection Issues

**Error**: SSH-related errors during command execution

**Solutions**:
1. Verify SSH key path in configuration file
2. Check SSH key permissions (should be 600)
3. Test manual SSH connection:
   ```bash
   ssh -i ~/.ssh/your_key user@host
   ```
4. Ensure firewall allows SSH connections
5. Verify server is reachable: `ping your.server.ip`

### Permission Denied Errors

**Error**: Permission denied when accessing files

**Solutions**:
1. Check SSH user has proper permissions on remote server
2. Verify the remote paths exist and are writable
3. Ensure backup directory exists or can be created

### Service Restart Timeout

**Warning**: `xkeen -restart is taking longer than 30 seconds`

**Explanation**: This is often normal behavior. The service may be restarting in the background.

**Action**: 
- Check service status: `xray-domain-updater status`
- If service is not running, restore from backup

### Invalid JSON Configuration

**Error**: `JSON parsing error` or `JSON structure validation error`

**Solutions**:
1. Validate JSON syntax using online validator or:
   ```bash
   python3 -m json.tool config.json
   ```
2. Ensure all required fields are present
3. Check for trailing commas or syntax errors
4. Restore from `config.example.json` if needed

### Missing Required Fields

**Error**: `Missing required field in configuration file`

**Solution**: Compare your configuration with `config.example.json` and ensure all sections are present:
- ssh (host, user, key_path)
- xray (remote_config_path, target_outbound_tag)
- backup (dir, count)

### Service Status Check Fails

**Error**: Service unavailable or configuration errors detected

**Actions**:
1. Check xkeen service logs on the server
2. Verify Xray configuration syntax
3. Restore from last known good backup:
   ```bash
   xray-domain-updater backup list
   xray-domain-updater backup restore 1 --confirm
   ```

## Uninstall

To remove the utility:

```bash
make uninstall
```

This will remove:
- Symlink from `~/.local/bin/xray-domain-updater`
- Configuration directory `~/.config/xray-domain-updater/`

Or manually:

```bash
rm ~/.local/bin/xray-domain-updater
rm -rf ~/.config/xray-domain-updater
```

## License

This project is provided as-is for personal and commercial use.

