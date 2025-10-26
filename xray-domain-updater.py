#!/usr/bin/env python3

import argparse
import functools
import json
import logging
import re
import subprocess
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


XKEEN_PATH = '/opt/sbin/xkeen'
XKEEN_ENV_PATH = 'PATH=/opt/bin:/opt/sbin:/bin:/sbin:/usr/bin:/usr/sbin'
XKEEN_RESTART_TIMEOUT = 30  # seconds
XKEEN_STATUS_TIMEOUT = 15  # seconds


BACKUP_FILENAME_FORMAT = 'routing_%Y%m%d_%H%M%S.json'
BACKUP_FILENAME_GLOB = 'routing_*.json'
BACKUP_FILENAME_REGEX = r'routing_(\d{8}_\d{6})\.json'
BACKUP_TIMESTAMP_FORMAT = '%Y%m%d_%H%M%S'


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Set up logging for the application."""
    logger = logging.getLogger('xray-updater')
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.propagate = False
    logger.handlers.clear()

    stdout = logging.StreamHandler(sys.stdout)
    stdout.addFilter(lambda r: r.levelno < logging.WARNING)
    logger.addHandler(stdout)

    stderr = logging.StreamHandler(sys.stderr)
    stderr.setLevel(logging.WARNING)
    logger.addHandler(stderr)

    return logger


def get_logger() -> logging.Logger:
    """Get configured logger."""
    return logging.getLogger('xray-updater')


@dataclass
class XrayRule:
    """Xray routing rule."""

    inbound_tag: list[str]
    """List of inbound connection tags."""
    outbound_tag: str
    """Outbound connection tag."""
    type: str
    """Routing rule type."""
    domain: list[str] | None = None
    """List of domains for routing (optional)."""
    network: str | None = None
    """Network type (optional)."""
    port: str | None = None
    """Port or port range (optional)."""
    protocol: list[str] | None = None
    """List of protocols (optional)."""


@dataclass
class XrayRouting:
    """Xray routing configuration."""

    domain_strategy: str
    """Domain name resolution strategy."""
    rules: list[XrayRule]
    """List of routing rules."""


@dataclass
class XrayConfig:
    """Xray configuration."""

    routing: XrayRouting
    """Routing configuration."""

    @classmethod
    def from_dict(cls, data: dict) -> 'XrayConfig':
        routing_data = data.get('routing', {})
        rules_data = routing_data.get('rules', [])

        rules = []
        for rule_data in rules_data:
            rule = XrayRule(
                inbound_tag=rule_data['inboundTag'],
                outbound_tag=rule_data['outboundTag'],
                type=rule_data.get('type', ''),
                domain=rule_data.get('domain'),
                network=rule_data.get('network'),
                port=rule_data.get('port'),
                protocol=rule_data.get('protocol'),
            )
            rules.append(rule)

        routing = XrayRouting(domain_strategy=routing_data['domainStrategy'], rules=rules)

        return cls(routing=routing)

    def to_dict(self) -> dict:
        rules_list = []
        for rule in self.routing.rules:
            rule_dict = {
                'inboundTag': rule.inbound_tag,
                'outboundTag': rule.outbound_tag,
                'type': rule.type,
            }
            if rule.domain is not None:
                rule_dict['domain'] = rule.domain
            if rule.network is not None:
                rule_dict['network'] = rule.network
            if rule.port is not None:
                rule_dict['port'] = rule.port
            if rule.protocol is not None:
                rule_dict['protocol'] = rule.protocol
            rules_list.append(rule_dict)

        return {
            'routing': {
                'domainStrategy': self.routing.domain_strategy,
                'rules': rules_list,
            }
        }

    def validate_json_structure(self) -> None:
        """Validate configuration structure before writing."""
        if not self.routing:
            raise ValueError('Missing routing section')
        if not self.routing.domain_strategy:
            raise ValueError('Missing domainStrategy')
        if not self.routing.rules:
            raise ValueError('Missing rules list')

        try:
            json.dumps(self.to_dict())
        except (TypeError, ValueError) as e:
            raise ValueError(f'Configuration cannot be serialized to JSON: {e}') from e

    def find_rule_by_outbound_tag(self, tag: str) -> XrayRule:
        for rule in self.routing.rules:
            if rule.outbound_tag == tag:
                return rule
        raise ValueError(f"Rule with outboundTag '{tag}' not found")

    def validate_rule_has_domains(self, rule: XrayRule) -> None:
        if rule.domain is None:
            raise ValueError(f"Rule with outbound_tag '{rule.outbound_tag}' does not contain domain field")
        if not isinstance(rule.domain, list):
            raise TypeError(f"Domain field in rule '{rule.outbound_tag}' is not a list")

    def get_domains_for_outbound(self, tag: str) -> list[str]:
        rule = self.find_rule_by_outbound_tag(tag)
        self.validate_rule_has_domains(rule)
        assert rule.domain is not None
        return rule.domain.copy()

    def add_domain_to_outbound(self, tag: str, domain: str) -> bool:
        rule = self.find_rule_by_outbound_tag(tag)
        self.validate_rule_has_domains(rule)
        assert rule.domain is not None

        if domain not in rule.domain:
            rule.domain.append(domain)
            return True
        return False

    def remove_domain_from_outbound(self, tag: str, domain: str) -> bool:
        rule = self.find_rule_by_outbound_tag(tag)
        self.validate_rule_has_domains(rule)
        assert rule.domain is not None

        if domain in rule.domain:
            rule.domain.remove(domain)
            return True
        return False


@dataclass
class BackupManager:
    """Manager for configuration backups."""

    backup_dir: str
    """Path to backups directory on remote server."""
    max_backups: int
    """Maximum number of backups to keep."""

    def create_backup_filename(self) -> str:
        """Create backup filename with current timestamp."""
        return datetime.now(UTC).strftime(BACKUP_FILENAME_FORMAT)

    def ensure_backup_dir_exists(self, *, config: 'ConfigFile') -> None:
        """Ensure backup directory exists on remote server."""
        try:
            execute_ssh_command(command=f'mkdir -p {self.backup_dir}', config=config, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            get_logger().error(f'Error creating backup directory: {e.stderr}')
            raise

    def create_backup(self, *, xray_config: XrayConfig, config: 'ConfigFile') -> str:
        """Create backup of current configuration."""
        backup_filename = self.create_backup_filename()
        backup_path = f'{self.backup_dir}/{backup_filename}'

        try:
            with tempfile.NamedTemporaryFile('w+', delete=False) as tmp:
                json.dump(xray_config.to_dict(), tmp, indent=2)
                tmp_path = tmp.name

            with open(tmp_path, 'rb') as f:
                execute_ssh_command(command=f'cat > {backup_path}', config=config, stdin=f, check=True)
            Path(tmp_path).unlink()
        except subprocess.CalledProcessError as e:
            get_logger().error(f'Error creating backup: {e.stderr}')
            raise
        else:
            return backup_filename

    def list_backups(self, *, config: 'ConfigFile') -> list[tuple[int, str, str]]:
        """Get list of backups: [(index, filename, timestamp)]."""
        try:
            result = execute_ssh_command(
                command=f'ls -1t {self.backup_dir}/{BACKUP_FILENAME_GLOB} 2>/dev/null || true',
                config=config,
                capture_output=True,
                check=True,
                text=True,
            )

            backups = []
            for i, line in enumerate(result.stdout.strip().split('\n'), 1):
                if line:
                    filename = Path(line).name
                    match = re.search(BACKUP_FILENAME_REGEX, filename)
                    if match:
                        timestamp_str = match.group(1)
                        try:
                            timestamp = datetime.strptime(timestamp_str, BACKUP_TIMESTAMP_FORMAT).replace(tzinfo=UTC)
                            readable_time = timestamp.strftime('%Y-%m-%d %H:%M:%S')
                            backups.append((i, filename, readable_time))
                        except ValueError:
                            backups.append((i, filename, 'unknown'))

        except subprocess.CalledProcessError:
            return []
        else:
            return backups

    def restore_backup(self, *, index: int, config: 'ConfigFile', preview_only: bool = False) -> str | None:
        """Restore backup by index."""
        backups = self.list_backups(config=config)
        if not backups or index < 1 or index > len(backups):
            raise ValueError(f'Invalid backup index: {index}')

        _, filename, _ = backups[index - 1]
        backup_path = f'{self.backup_dir}/{filename}'

        try:
            result = execute_ssh_command(
                command=f'cat {backup_path}', config=config, capture_output=True, check=True, text=True
            )

            if preview_only:
                return result.stdout

            execute_ssh_command(command=f'cp {backup_path} {config.remote_json_path}', config=config, check=True)
        except subprocess.CalledProcessError as e:
            get_logger().error(f'Error restoring backup: {e.stderr}')
            raise
        else:
            return None

    def cleanup_old_backups(self, *, config: 'ConfigFile') -> None:
        """Remove old backups, keeping only max_backups."""
        backups = self.list_backups(config=config)
        if len(backups) <= self.max_backups:
            return

        to_remove = backups[self.max_backups :]

        try:
            for _, filename, _ in to_remove:
                backup_path = f'{self.backup_dir}/{filename}'
                execute_ssh_command(command=f'rm -f {backup_path}', config=config, check=True)
        except subprocess.CalledProcessError as e:
            get_logger().warning(f'Error cleaning up old backups: {e.stderr}')


@dataclass
class ConfigFile:
    """Connection and application parameters configuration."""

    ssh_host: str
    """Remote server address."""
    ssh_user: str
    """Username for SSH connection."""
    ssh_key: Path
    """Path to private SSH key."""
    remote_json_path: str
    """Path to Xray configuration file on remote server."""
    backup_dir: str
    """Directory for storing backups on remote server."""
    backup_count: int
    """Maximum number of backups to keep."""
    target_outbound_tag: str
    """Outbound tag for domain management."""

    @staticmethod
    def get_config_path() -> Path:
        """Return configuration file path with cascading search.

        Checks in the following order:
        1. Next to script: ./config.json
        2. In XDG directory: ~/.config/xray-domain-updater/config.json

        Returns first found file or XDG path if nothing found.
        """
        script_dir_config = Path(__file__).parent / 'config.json'
        if script_dir_config.exists():
            return script_dir_config

        # Path in XDG directory (fixed name: xray-domain-updater)
        xdg_config = Path.home() / '.config' / 'xray-domain-updater' / 'config.json'
        if xdg_config.exists():
            return xdg_config

        # If nothing found, return XDG path for error message
        return xdg_config

    @staticmethod
    def check_file_permissions(path: Path) -> None:
        """Check file permissions (should be 600 or 400)."""
        if not path.exists():
            return

        stat_info = path.stat()
        mode = stat_info.st_mode & 0o777

        if mode not in [0o600, 0o400]:
            get_logger().warning(f'WARNING: Unsafe permissions on configuration file: {oct(mode)}')
            get_logger().warning(f'Recommended: chmod 600 {path}')

    @classmethod
    def from_file(cls, config_path: Path | None = None) -> 'ConfigFile':
        """Load configuration from file."""
        if config_path is None:
            config_path = cls.get_config_path()

        if not config_path.exists():
            get_logger().error('Configuration file not found')
            get_logger().error('Checked the following locations:')

            script_dir_config = Path(__file__).parent / 'config.json'
            xdg_config = Path.home() / '.config' / 'xray-domain-updater' / 'config.json'

            get_logger().error(f'  1. {script_dir_config}')
            get_logger().error(f'  2. {xdg_config}')

            example_path = Path(__file__).parent / 'config.example.json'
            if example_path.exists():
                get_logger().error(f'\nExample configuration: {example_path}')
            else:
                get_logger().error('\nExample configuration not available')

            get_logger().error('Create a configuration file and fill it with data')
            get_logger().error('Or use --config flag to specify the file path')
            sys.exit(1)

        cls.check_file_permissions(config_path)

        try:
            with open(config_path) as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            get_logger().error(f'JSON parsing error in configuration file: {e}')
            sys.exit(1)
        except Exception as e:
            get_logger().error(f'Error reading configuration file: {e}')
            sys.exit(1)

        try:
            ssh_config = data['ssh']
            xray_config = data['xray']
            backup_config = data['backup']

            ssh_key_path = Path(ssh_config['key_path']).expanduser()

            config = cls(
                ssh_host=ssh_config['host'],
                ssh_user=ssh_config['user'],
                ssh_key=ssh_key_path,
                remote_json_path=xray_config['remote_config_path'],
                backup_dir=backup_config['dir'],
                backup_count=backup_config['count'],
                target_outbound_tag=xray_config['target_outbound_tag'],
            )

            config.validate()
        except KeyError as e:
            get_logger().error(f'Missing required field in configuration file: {e}')
            sys.exit(1)
        else:
            return config

    def validate(self) -> None:
        """Validate configuration parameters."""
        if not self.ssh_key.exists():
            get_logger().error(f'SSH key not found: {self.ssh_key}')
            get_logger().error('Check key path in configuration file')
            sys.exit(1)

        self.check_file_permissions(self.ssh_key)


def execute_ssh_command(
    *, command: str, config: ConfigFile, check: bool = False, **kwargs: Any
) -> subprocess.CompletedProcess[Any]:
    """Execute SSH command on remote server."""
    ssh_cmd = [
        'ssh',
        '-i',
        str(config.ssh_key),
        f'{config.ssh_user}@{config.ssh_host}',
        command,
    ]
    return subprocess.run(ssh_cmd, check=check, **kwargs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Manage xray domain list on remote server.')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output')
    parser.add_argument(
        '--config',
        type=Path,
        default=None,
        help='Path to configuration file (default: ./config.json or ~/.config/xray-domain-updater/config.json)',
    )
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    subparsers.add_parser('list', help='Show domain list')

    add_parser = subparsers.add_parser('add', help='Add domain')
    add_parser.add_argument('domain', help='Domain to add')

    remove_parser = subparsers.add_parser('remove', help='Remove domain by index')
    remove_parser.add_argument('index', type=int, help='Domain index to remove (from list command)')
    remove_parser.add_argument('-y', '--yes', action='store_true', help='Confirm removal without prompt')

    subparsers.add_parser('restart', help='Restart service')

    backup_parser = subparsers.add_parser('backup', help='Backup management')
    backup_subparsers = backup_parser.add_subparsers(dest='backup_command')

    backup_subparsers.add_parser('list', help='Show backup list')
    backup_subparsers.add_parser('clean', help='Clean old backups')

    restore_parser = backup_subparsers.add_parser('restore', help='Restore from backup')
    restore_parser.add_argument('index', type=int, help='Backup index to restore')
    restore_parser.add_argument('--preview', action='store_true', help='Show backup content')
    restore_parser.add_argument('--confirm', action='store_true', help='Confirm restoration')

    subparsers.add_parser('status', help='Check service status')

    return parser.parse_args()


def read_remote_json(config: ConfigFile) -> XrayConfig:
    """Read JSON configuration from remote server."""
    try:
        result = execute_ssh_command(
            command=f'cat {config.remote_json_path}',
            config=config,
            check=True,
            capture_output=True,
            text=True,
        )
        return XrayConfig.from_dict(json.loads(result.stdout))
    except subprocess.CalledProcessError as e:
        get_logger().error(f'Error reading JSON file {config.remote_json_path}: {e.stderr}')
        sys.exit(1)
    except json.JSONDecodeError as e:
        get_logger().error(f'JSON parsing error: {e}')
        sys.exit(1)
    except ValueError as e:
        get_logger().error(f'JSON structure validation error: {e}')
        sys.exit(1)


def write_remote_json(*, xray_config: XrayConfig, config: ConfigFile, create_backup: bool = True) -> None:
    """Write JSON configuration to remote server with backup and validation."""
    xray_config.validate_json_structure()

    if create_backup:
        backup_manager = BackupManager(config.backup_dir, config.backup_count)
        backup_manager.ensure_backup_dir_exists(config=config)
        current_config = read_remote_json(config)
        backup_filename = backup_manager.create_backup(xray_config=current_config, config=config)
        get_logger().info(f'Backup created: {backup_filename}')

    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile('w+', delete=False) as tmp:
            json.dump(xray_config.to_dict(), tmp, indent=2)
            tmp_path = tmp.name

        with open(tmp_path, 'rb') as f:
            execute_ssh_command(command=f'cat > {config.remote_json_path}', config=config, stdin=f, check=True)
        get_logger().info(f'JSON successfully uploaded to server {config.ssh_host}')
    except subprocess.CalledProcessError as e:
        get_logger().error(f'Error writing JSON file: {e.stderr}')
        sys.exit(1)
    finally:
        if tmp_path and Path(tmp_path).exists():
            Path(tmp_path).unlink()

    if create_backup:
        backup_manager.cleanup_old_backups(config=config)


def restart_service(config: ConfigFile) -> None:
    """Restart xkeen service with proper PATH to avoid missing utilities."""
    try:
        get_logger().info('Executing xkeen -restart')

        restart_command = f'{XKEEN_ENV_PATH} {XKEEN_PATH} -restart 2>&1'

        execute_ssh_command(
            command=restart_command, config=config, check=False, timeout=XKEEN_RESTART_TIMEOUT, text=True
        )
        get_logger().info('xkeen -restart completed successfully')

    except subprocess.TimeoutExpired:
        get_logger().info('xkeen -restart is taking longer than 30 seconds')
        get_logger().info('This may be normal - service is restarting in background')
        get_logger().info('Restart command sent to server')
    except subprocess.CalledProcessError as e:
        get_logger().error(f'Error executing xkeen -restart: return code {e.returncode}')
        if hasattr(e, 'stderr') and e.stderr:
            get_logger().error(f'stderr: {e.stderr}')
    except Exception as e:
        get_logger().error(f'Unexpected error during restart: {e}')

    get_logger().info('Checking service status after restart...')
    if check_service_status(config):
        get_logger().info('Service is running correctly')
    else:
        get_logger().error('WARNING: Service unavailable or configuration errors detected')
        get_logger().error('Use backup restore command to rollback')


def check_service_status(config: ConfigFile) -> bool:
    """Check xkeen service status with console output."""
    try:
        get_logger().info('Executing xkeen -status...')
        result = execute_ssh_command(
            command=f'{XKEEN_ENV_PATH} {XKEEN_PATH} -status',
            config=config,
            check=False,
            text=True,
            timeout=XKEEN_STATUS_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        get_logger().error('Timeout executing xkeen -status')
        return False
    except Exception as e:
        get_logger().error(f'Error executing xkeen -status: {e}')
        return False
    else:
        return result.returncode == 0


def list_domains(*, xray_config: XrayConfig, config: ConfigFile) -> None:
    """Display numbered list of domains with total count."""
    try:
        domain_list = xray_config.get_domains_for_outbound(config.target_outbound_tag)

        for i, domain in enumerate(domain_list, 1):
            get_logger().info(f'{i}. {domain}')

        get_logger().info(f'\nTotal domains: {len(domain_list)}')
    except ValueError as e:
        get_logger().error(f'{e}')
        sys.exit(1)


def add_domain_to_config(*, xray_config: XrayConfig, domain_to_add: str, config: ConfigFile) -> XrayConfig:
    """Add domain to the configuration."""
    try:
        added = xray_config.add_domain_to_outbound(config.target_outbound_tag, domain_to_add)
        if added:
            get_logger().info(f'Domain added: {domain_to_add}')
        else:
            get_logger().info(f'Domain {domain_to_add} already in list')
    except ValueError as e:
        get_logger().error(f'{e}')
        sys.exit(1)
    return xray_config


def remove_domain_from_config(
    *, xray_config: XrayConfig, domain_index: int, config: ConfigFile, confirm: bool = False
) -> XrayConfig:
    """Remove domain from the configuration by index."""
    try:
        domains = xray_config.get_domains_for_outbound(config.target_outbound_tag)

        if domain_index < 1 or domain_index > len(domains):
            get_logger().error(f'Invalid index {domain_index}. Available indices: 1-{len(domains)}')
            sys.exit(1)

        domain_to_remove = domains[domain_index - 1]

        if not confirm:
            get_logger().info(f'Domain to remove: [{domain_index}] {domain_to_remove}')
            get_logger().info('Use -y flag to confirm removal')
            return xray_config

        xray_config.remove_domain_from_outbound(config.target_outbound_tag, domain_to_remove)
        get_logger().info(f'Domain [{domain_index}] {domain_to_remove} successfully removed')
    except ValueError as e:
        get_logger().error(f'{e}')
        sys.exit(1)
    return xray_config


def command_handler(func: Callable[[argparse.Namespace, ConfigFile], None]) -> Callable[[argparse.Namespace], None]:
    """Decorator for command handlers with config loading and error handling."""

    @functools.wraps(func)
    def wrapper(args: argparse.Namespace) -> None:
        try:
            config = ConfigFile.from_file(args.config if hasattr(args, 'config') else None)
            return func(args, config)
        except (ValueError, KeyError, TypeError) as e:
            command_name = func.__name__.replace('handle_', '').replace('_command', '')
            get_logger().error(f'Error handling {command_name} command: {e}')
            sys.exit(1)

    return wrapper


@command_handler
def handle_list_command(args: argparse.Namespace, config: ConfigFile) -> None:
    """Handle list command."""
    xray_config: XrayConfig = read_remote_json(config)
    list_domains(xray_config=xray_config, config=config)


@command_handler
def handle_add_command(args: argparse.Namespace, config: ConfigFile) -> None:
    """Handle add command."""
    xray_config: XrayConfig = read_remote_json(config)
    updated: XrayConfig = add_domain_to_config(xray_config=xray_config, domain_to_add=args.domain, config=config)
    write_remote_json(xray_config=updated, config=config, create_backup=True)
    restart_service(config)


@command_handler
def handle_remove_command(args: argparse.Namespace, config: ConfigFile) -> None:
    """Handle remove command."""
    xray_config: XrayConfig = read_remote_json(config)

    confirm = getattr(args, 'yes', False)

    updated: XrayConfig = remove_domain_from_config(
        xray_config=xray_config, domain_index=args.index, config=config, confirm=confirm
    )

    if confirm:
        write_remote_json(xray_config=updated, config=config, create_backup=True)
        restart_service(config)


@command_handler
def handle_restart_command(args: argparse.Namespace, config: ConfigFile) -> None:
    """Handle restart command."""
    restart_service(config)


@command_handler
def handle_backup_list_command(args: argparse.Namespace, config: ConfigFile) -> None:
    """Handle backup list command."""
    backup_manager = BackupManager(config.backup_dir, config.backup_count)
    backups = backup_manager.list_backups(config=config)
    if not backups:
        get_logger().info('No backups found')
        return

    for index, filename, timestamp in backups:
        get_logger().info(f'{index}. {filename} ({timestamp})')
    get_logger().info(f'\nTotal backups: {len(backups)}')


@command_handler
def handle_backup_clean_command(args: argparse.Namespace, config: ConfigFile) -> None:
    """Handle backup clean command."""
    backup_manager = BackupManager(config.backup_dir, config.backup_count)
    backup_manager.cleanup_old_backups(config=config)
    get_logger().info('Old backups cleaned')


def handle_backup_command(args: argparse.Namespace) -> None:
    """Handle backup command routing."""
    if args.backup_command == 'list':
        handle_backup_list_command(args)
    elif args.backup_command == 'clean':
        handle_backup_clean_command(args)
    elif args.backup_command == 'restore':
        handle_restore_command(args)
    else:
        get_logger().error('Unknown backup command')
        sys.exit(1)


@command_handler
def handle_restore_command(args: argparse.Namespace, config: ConfigFile) -> None:
    """Handle restore command."""
    if not args.preview and not args.confirm:
        get_logger().error('Specify --preview to view or --confirm to restore')
        sys.exit(1)

    backup_manager = BackupManager(config.backup_dir, config.backup_count)

    if args.preview:
        content = backup_manager.restore_backup(index=args.index, config=config, preview_only=True)
        get_logger().info(f'Backup #{args.index} content:')
        if content:
            get_logger().info(content)

    if args.confirm:
        backup_manager.restore_backup(index=args.index, config=config, preview_only=False)
        get_logger().info(f'Backup #{args.index} restored')
        restart_service(config)


@command_handler
def handle_status_command(args: argparse.Namespace, config: ConfigFile) -> None:
    """Handle status command."""
    if check_service_status(config):
        get_logger().info('Service is running correctly')
    else:
        get_logger().error('Service unavailable or configuration errors detected')


COMMANDS: dict[str, Callable[[argparse.Namespace], None]] = {
    'list': handle_list_command,
    'add': handle_add_command,
    'remove': handle_remove_command,
    'restart': handle_restart_command,
    'backup': handle_backup_command,
    'status': handle_status_command,
}


def main() -> None:
    """Main application entry point."""
    args: argparse.Namespace = parse_args()
    setup_logging(args.verbose if hasattr(args, 'verbose') else False)

    if not args.command:
        get_logger().error('No command specified. Use -h or --help for assistance.')
        sys.exit(1)

    command_handler: Callable[[argparse.Namespace], None] | None = COMMANDS.get(args.command)
    if command_handler:
        command_handler(args)
    else:
        get_logger().error(f'Unknown command: {args.command}')
        sys.exit(1)


if __name__ == '__main__':
    main()
