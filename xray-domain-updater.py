#!/usr/bin/env python3

import argparse
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


# ==== Xkeen service configuration ====
XKEEN_PATH = '/opt/sbin/xkeen'
XKEEN_ENV_PATH = 'PATH=/opt/bin:/opt/sbin:/bin:/sbin:/usr/bin:/usr/sbin'
XKEEN_RESTART_TIMEOUT = 30  # seconds
XKEEN_STATUS_TIMEOUT = 15  # seconds


# ==== Backup filename configuration ====
BACKUP_FILENAME_FORMAT = 'routing_%Y%m%d_%H%M%S.json'
BACKUP_FILENAME_GLOB = 'routing_*.json'
BACKUP_FILENAME_REGEX = r'routing_(\d{8}_\d{6})\.json'
BACKUP_TIMESTAMP_FORMAT = '%Y%m%d_%H%M%S'


# ==== Logging configuration ====
def setup_logging(verbose: bool = False) -> logging.Logger:
    """Настроить logging для приложения."""
    logger = logging.getLogger('xray-updater')
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.propagate = False

    # Удаляем существующие обработчики
    logger.handlers.clear()

    # Console handler для INFO и DEBUG (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    console_handler.addFilter(lambda record: record.levelno < logging.WARNING)

    # Error handler для WARNING, ERROR, CRITICAL (stderr)
    error_handler = logging.StreamHandler(sys.stderr)
    error_handler.setLevel(logging.WARNING)

    logger.addHandler(console_handler)
    logger.addHandler(error_handler)

    return logger


def get_logger() -> logging.Logger:
    """Получить настроенный logger."""
    return logging.getLogger('xray-updater')


@dataclass
class XrayRule:
    inbound_tag: list[str]
    outbound_tag: str
    type: str
    domain: list[str] | None = None
    network: str | None = None
    port: str | None = None
    protocol: list[str] | None = None


@dataclass
class XrayRouting:
    domain_strategy: str
    rules: list[XrayRule]


@dataclass
class XrayConfig:
    routing: XrayRouting

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
        """Валидировать структуру конфигурации перед записью."""
        if not self.routing:
            raise ValueError('Отсутствует секция routing')
        if not self.routing.domain_strategy:
            raise ValueError('Отсутствует domainStrategy')
        if not self.routing.rules:
            raise ValueError('Отсутствует список правил')

        try:
            json.dumps(self.to_dict())
        except (TypeError, ValueError) as e:
            raise ValueError(f'Конфигурация не может быть сериализована в JSON: {e}') from e

    def find_rule_by_outbound_tag(self, tag: str) -> XrayRule:
        for rule in self.routing.rules:
            if rule.outbound_tag == tag:
                return rule
        raise ValueError(f"Правило с outboundTag '{tag}' не найдено")

    def validate_rule_has_domains(self, rule: XrayRule) -> None:
        if rule.domain is None:
            raise ValueError(f"Правило с outbound_tag '{rule.outbound_tag}' не содержит поля domain")
        if not isinstance(rule.domain, list):
            raise TypeError(f"Поле domain в правиле '{rule.outbound_tag}' не является списком")

    def get_domains_for_outbound(self, tag: str) -> list[str]:
        rule = self.find_rule_by_outbound_tag(tag)
        self.validate_rule_has_domains(rule)
        assert rule.domain is not None  # После validate_rule_has_domains это гарантированно
        return rule.domain.copy()

    def add_domain_to_outbound(self, tag: str, domain: str) -> bool:
        rule = self.find_rule_by_outbound_tag(tag)
        self.validate_rule_has_domains(rule)
        assert rule.domain is not None  # После validate_rule_has_domains это гарантированно

        if domain not in rule.domain:
            rule.domain.append(domain)
            return True
        return False

    def remove_domain_from_outbound(self, tag: str, domain: str) -> bool:
        rule = self.find_rule_by_outbound_tag(tag)
        self.validate_rule_has_domains(rule)
        assert rule.domain is not None  # После validate_rule_has_domains это гарантированно

        if domain in rule.domain:
            rule.domain.remove(domain)
            return True
        return False


@dataclass
class BackupManager:
    backup_dir: str
    max_backups: int

    def create_backup_filename(self) -> str:
        """Создать имя файла бэкапа с текущим timestamp."""
        return datetime.now(UTC).strftime(BACKUP_FILENAME_FORMAT)

    def ensure_backup_dir_exists(self, config: 'ConfigFile') -> None:
        """Убедиться что директория бэкапов существует на удаленном сервере."""
        try:
            execute_ssh_command(f'mkdir -p {self.backup_dir}', config, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            get_logger().error(f'Ошибка создания директории бэкапов: {e.stderr}')
            raise

    def create_backup(self, xray_config: XrayConfig, config: 'ConfigFile') -> str:
        """Создать бэкап текущей конфигурации."""
        backup_filename = self.create_backup_filename()
        backup_path = f'{self.backup_dir}/{backup_filename}'

        try:
            with tempfile.NamedTemporaryFile('w+', delete=False) as tmp:
                json.dump(xray_config.to_dict(), tmp, indent=2)
                tmp_path = tmp.name

            with open(tmp_path, 'rb') as f:
                execute_ssh_command(f'cat > {backup_path}', config, stdin=f, check=True)
            Path(tmp_path).unlink()
        except subprocess.CalledProcessError as e:
            get_logger().error(f'Ошибка создания бэкапа: {e.stderr}')
            raise
        else:
            return backup_filename

    def list_backups(self, config: 'ConfigFile') -> list[tuple[int, str, str]]:
        """Получить список бэкапов: [(index, filename, timestamp)]."""
        try:
            result = execute_ssh_command(
                f'ls -1t {self.backup_dir}/{BACKUP_FILENAME_GLOB} 2>/dev/null || true',
                config,
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
                            backups.append((i, filename, 'неизвестно'))

        except subprocess.CalledProcessError:
            return []
        else:
            return backups

    def restore_backup(self, index: int, config: 'ConfigFile', preview_only: bool = False) -> str | None:
        """Восстановить бэкап по индексу."""
        backups = self.list_backups(config)
        if not backups or index < 1 or index > len(backups):
            raise ValueError(f'Неверный индекс бэкапа: {index}')

        _, filename, _ = backups[index - 1]
        backup_path = f'{self.backup_dir}/{filename}'

        try:
            result = execute_ssh_command(f'cat {backup_path}', config, capture_output=True, check=True, text=True)

            if preview_only:
                return result.stdout

            execute_ssh_command(f'cp {backup_path} {config.remote_json_path}', config, check=True)
        except subprocess.CalledProcessError as e:
            get_logger().error(f'Ошибка восстановления бэкапа: {e.stderr}')
            raise
        else:
            return None

    def cleanup_old_backups(self, config: 'ConfigFile') -> None:
        """Удалить старые бэкапы, оставив только max_backups."""
        backups = self.list_backups(config)
        if len(backups) <= self.max_backups:
            return

        to_remove = backups[self.max_backups :]

        try:
            for _, filename, _ in to_remove:
                backup_path = f'{self.backup_dir}/{filename}'
                execute_ssh_command(f'rm -f {backup_path}', config, check=True)
        except subprocess.CalledProcessError as e:
            get_logger().warning(f'Ошибка очистки старых бэкапов: {e.stderr}')


@dataclass
class ConfigFile:
    ssh_host: str
    ssh_user: str
    ssh_key: Path
    remote_json_path: str
    backup_dir: str
    backup_count: int
    target_outbound_tag: str

    @staticmethod
    def get_config_path() -> Path:
        """Возвращает путь к конфигурационному файлу с каскадным поиском.

        Проверяет в следующем порядке:
        1. Рядом со скриптом: ./config.json
        2. В XDG директории: ~/.config/xray-domain-updater/config.json

        Возвращает первый найденный файл или XDG путь если ничего не найдено.
        """
        # Путь рядом со скриптом
        script_dir_config = Path(__file__).parent / 'config.json'
        if script_dir_config.exists():
            return script_dir_config

        # Путь в XDG директории (исправлено имя: xray-domain-updater)
        xdg_config = Path.home() / '.config' / 'xray-domain-updater' / 'config.json'
        if xdg_config.exists():
            return xdg_config

        # Если ничего не найдено, возвращаем XDG путь для сообщения об ошибке
        return xdg_config

    @staticmethod
    def check_file_permissions(path: Path) -> None:
        """Проверяет права доступа к файлу (должны быть 600 или 400)."""
        if not path.exists():
            return

        stat_info = path.stat()
        mode = stat_info.st_mode & 0o777

        if mode not in [0o600, 0o400]:
            get_logger().warning(f'ПРЕДУПРЕЖДЕНИЕ: Небезопасные права доступа к конфигурационному файлу: {oct(mode)}')
            get_logger().warning(f'Рекомендуется изменить права доступа: chmod 600 {path}')

    @classmethod
    def from_file(cls, config_path: Path | None = None) -> 'ConfigFile':
        """Загружает конфигурацию из файла."""
        if config_path is None:
            config_path = cls.get_config_path()

        if not config_path.exists():
            get_logger().error('Конфигурационный файл не найден')
            get_logger().error('Проверены следующие местоположения:')

            script_dir_config = Path(__file__).parent / 'config.json'
            xdg_config = Path.home() / '.config' / 'xray-domain-updater' / 'config.json'

            get_logger().error(f'  1. {script_dir_config}')
            get_logger().error(f'  2. {xdg_config}')

            example_path = Path(__file__).parent / 'config.example.json'
            if example_path.exists():
                get_logger().error(f'\nПример конфигурации: {example_path}')
            else:
                get_logger().error('\nПример конфигурации недоступен')

            get_logger().error('Создайте конфигурационный файл и заполните его данными')
            get_logger().error('Или используйте флаг --config для указания пути к файлу')
            sys.exit(1)

        cls.check_file_permissions(config_path)

        try:
            with open(config_path) as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            get_logger().error(f'Ошибка парсинга JSON в конфигурационном файле: {e}')
            sys.exit(1)
        except Exception as e:
            get_logger().error(f'Ошибка чтения конфигурационного файла: {e}')
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
            get_logger().error(f'Отсутствует обязательное поле в конфигурационном файле: {e}')
            sys.exit(1)
        else:
            return config

    def validate(self) -> None:
        """Валидирует параметры конфигурации."""
        if not self.ssh_key.exists():
            get_logger().error(f'SSH ключ не найден: {self.ssh_key}')
            get_logger().error('Проверьте путь к ключу в конфигурационном файле')
            sys.exit(1)

        self.check_file_permissions(self.ssh_key)


def execute_ssh_command(
    command: str, config: ConfigFile, *, check: bool = False, **kwargs: Any
) -> subprocess.CompletedProcess[Any]:
    """Выполнить SSH команду на удаленном сервере."""
    ssh_cmd = [
        'ssh',
        '-i',
        str(config.ssh_key),
        f'{config.ssh_user}@{config.ssh_host}',
        command,
    ]
    return subprocess.run(ssh_cmd, check=check, **kwargs)


# ==== Argument parsing ====
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Управление списком доменов xray на удалённом сервере.')
    parser.add_argument('--verbose', '-v', action='store_true', help='Включить подробный вывод')
    parser.add_argument(
        '--config',
        type=Path,
        default=None,
        help='Путь к конфигурационному файлу '
        '(по умолчанию: ./config.json или ~/.config/xray-domain-updater/config.json)',
    )
    subparsers = parser.add_subparsers(dest='command', help='Доступные команды')

    # Существующие команды
    subparsers.add_parser('list', help='Показать список доменов')

    add_parser = subparsers.add_parser('add', help='Добавить домен')
    add_parser.add_argument('domain', help='Домен для добавления')

    remove_parser = subparsers.add_parser('remove', help='Удалить домен по индексу')
    remove_parser.add_argument('index', type=int, help='Индекс домена для удаления (из списка list)')
    remove_parser.add_argument('-y', '--yes', action='store_true', help='Подтвердить удаление без запроса')

    subparsers.add_parser('restart', help='Перезапустить сервис')

    # Новые команды для бэкапов
    backup_parser = subparsers.add_parser('backup', help='Управление бэкапами')
    backup_subparsers = backup_parser.add_subparsers(dest='backup_command')

    backup_subparsers.add_parser('list', help='Показать список бэкапов')
    backup_subparsers.add_parser('clean', help='Очистить старые бэкапы')

    restore_parser = backup_subparsers.add_parser('restore', help='Восстановить из бэкапа')
    restore_parser.add_argument('index', type=int, help='Индекс бэкапа для восстановления')
    restore_parser.add_argument('--preview', action='store_true', help='Показать содержимое бэкапа')
    restore_parser.add_argument('--confirm', action='store_true', help='Подтвердить восстановление')

    subparsers.add_parser('status', help='Проверить статус сервиса')

    return parser.parse_args()


# ==== JSON operations ====
def read_remote_json(config: ConfigFile) -> XrayConfig:
    """Read JSON configuration from remote server."""
    try:
        result = execute_ssh_command(
            f'cat {config.remote_json_path}',
            config,
            check=True,
            capture_output=True,
            text=True,
        )
        return XrayConfig.from_dict(json.loads(result.stdout))
    except subprocess.CalledProcessError as e:
        get_logger().error(f'Ошибка при чтении JSON файла {config.remote_json_path}: {e.stderr}')
        sys.exit(1)
    except json.JSONDecodeError as e:
        get_logger().error(f'Ошибка парсинга JSON: {e}')
        sys.exit(1)
    except ValueError as e:
        get_logger().error(f'Ошибка валидации структуры JSON: {e}')
        sys.exit(1)


def write_remote_json(xray_config: XrayConfig, config: ConfigFile, create_backup: bool = True) -> None:
    """Write JSON configuration to remote server with backup and validation."""
    xray_config.validate_json_structure()

    if create_backup:
        backup_manager = BackupManager(config.backup_dir, config.backup_count)
        backup_manager.ensure_backup_dir_exists(config)
        current_config = read_remote_json(config)
        backup_filename = backup_manager.create_backup(current_config, config)
        get_logger().info(f'Создан бэкап: {backup_filename}')

    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile('w+', delete=False) as tmp:
            json.dump(xray_config.to_dict(), tmp, indent=2)
            tmp_path = tmp.name

        with open(tmp_path, 'rb') as f:
            execute_ssh_command(f'cat > {config.remote_json_path}', config, stdin=f, check=True)
        get_logger().info(f'JSON успешно отправлен на сервер {config.ssh_host}')
    except subprocess.CalledProcessError as e:
        get_logger().error(f'Ошибка при записи JSON файла: {e.stderr}')
        sys.exit(1)
    finally:
        if tmp_path and Path(tmp_path).exists():
            Path(tmp_path).unlink()

    if create_backup:
        backup_manager.cleanup_old_backups(config)


def restart_service(config: ConfigFile) -> None:
    """Restart xkeen service with proper PATH to avoid missing utilities."""
    try:
        get_logger().info('Выполняется xkeen -restart')

        restart_command = f'{XKEEN_ENV_PATH} {XKEEN_PATH} -restart 2>&1'

        execute_ssh_command(restart_command, config, check=False, timeout=XKEEN_RESTART_TIMEOUT, text=True)
        get_logger().info('xkeen -restart завершен успешно')

    except subprocess.TimeoutExpired:
        get_logger().info('xkeen -restart выполняется дольше 30 секунд')
        get_logger().info('Это может быть нормально - сервис перезапускается в фоне')
        get_logger().info('Команда restart отправлена на сервер')
    except subprocess.CalledProcessError as e:
        get_logger().error(f'Ошибка при выполнении xkeen -restart: код возврата {e.returncode}')
        if hasattr(e, 'stderr') and e.stderr:
            get_logger().error(f'stderr: {e.stderr}')
    except Exception as e:
        get_logger().error(f'Неожиданная ошибка при перезапуске: {e}')

    get_logger().info('Проверка статуса сервиса после перезапуска...')
    if check_service_status_with_output(config):
        get_logger().info('Сервис работает корректно')
    else:
        get_logger().error('ВНИМАНИЕ: Сервис недоступен или есть ошибки конфигурации')
        get_logger().error('Используйте команду backup restore для отката')


def check_service_status(config: ConfigFile) -> bool:
    """Проверить статус сервиса xkeen."""
    try:
        execute_ssh_command(
            f'{XKEEN_ENV_PATH} {XKEEN_PATH} -status',
            config,
            check=True,
            capture_output=True,
            text=True,
            timeout=XKEEN_STATUS_TIMEOUT,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return False
    else:
        return True


def check_service_status_with_output(config: ConfigFile) -> bool:
    """Проверить статус сервиса xkeen с выводом в консоль."""
    try:
        get_logger().info('Выполнение xkeen -status...')
        result = execute_ssh_command(
            f'{XKEEN_ENV_PATH} {XKEEN_PATH} -status',
            config,
            check=False,
            text=True,
            timeout=XKEEN_STATUS_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        get_logger().error('Тайм-аут при выполнении xkeen -status')
        return False
    except Exception as e:
        get_logger().error(f'Ошибка при выполнении xkeen -status: {e}')
        return False
    else:
        return result.returncode == 0


# ==== Domain management functions ====
def list_domains(xray_config: XrayConfig, config: ConfigFile) -> None:
    """Display numbered list of domains with total count."""
    try:
        domain_list = xray_config.get_domains_for_outbound(config.target_outbound_tag)

        for i, domain in enumerate(domain_list, 1):
            get_logger().info(f'{i}. {domain}')

        get_logger().info(f'\nВсего доменов: {len(domain_list)}')
    except ValueError as e:
        get_logger().error(f'{e}')
        sys.exit(1)


def add_domain_to_config(xray_config: XrayConfig, domain_to_add: str, config: ConfigFile) -> XrayConfig:
    """Add domain to the configuration."""
    try:
        added = xray_config.add_domain_to_outbound(config.target_outbound_tag, domain_to_add)
        if added:
            get_logger().info(f'Добавлен домен: {domain_to_add}')
        else:
            get_logger().info(f'Домен {domain_to_add} уже есть в списке')
    except ValueError as e:
        get_logger().error(f'{e}')
        sys.exit(1)
    return xray_config


def remove_domain_from_config(
    xray_config: XrayConfig, domain_index: int, config: ConfigFile, confirm: bool = False
) -> XrayConfig:
    """Remove domain from the configuration by index."""
    try:
        domains = xray_config.get_domains_for_outbound(config.target_outbound_tag)

        if domain_index < 1 or domain_index > len(domains):
            get_logger().error(f'Неверный индекс {domain_index}. Доступные индексы: 1-{len(domains)}')
            sys.exit(1)

        domain_to_remove = domains[domain_index - 1]

        if not confirm:
            # Режим preview: только показать информацию
            get_logger().info(f'Домен для удаления: [{domain_index}] {domain_to_remove}')
            get_logger().info('Для подтверждения удаления используйте флаг -y')
            return xray_config  # Возвращаем без изменений

        # Режим confirm: удалить домен
        xray_config.remove_domain_from_outbound(config.target_outbound_tag, domain_to_remove)
        get_logger().info(f'Домен [{domain_index}] {domain_to_remove} успешно удален')
    except ValueError as e:
        get_logger().error(f'{e}')
        sys.exit(1)
    return xray_config


# ==== Command handlers ====
def handle_list_command(args: argparse.Namespace) -> None:
    """Handle list command."""
    try:
        config = ConfigFile.from_file(args.config if hasattr(args, 'config') else None)
        xray_config: XrayConfig = read_remote_json(config)
        list_domains(xray_config, config)
    except (ValueError, KeyError, TypeError) as e:
        get_logger().error(f'Ошибка при обработке команды list: {e}')
        sys.exit(1)


def handle_add_command(args: argparse.Namespace) -> None:
    """Handle add command."""
    try:
        config = ConfigFile.from_file(args.config if hasattr(args, 'config') else None)
        xray_config: XrayConfig = read_remote_json(config)
        updated: XrayConfig = add_domain_to_config(xray_config, args.domain, config)
        write_remote_json(updated, config, create_backup=True)
        restart_service(config)
    except (ValueError, KeyError, TypeError) as e:
        get_logger().error(f'Ошибка при обработке команды add: {e}')
        sys.exit(1)


def handle_remove_command(args: argparse.Namespace) -> None:
    """Handle remove command."""
    try:
        config = ConfigFile.from_file(args.config if hasattr(args, 'config') else None)
        xray_config: XrayConfig = read_remote_json(config)

        # Проверяем наличие флага -y
        confirm = getattr(args, 'yes', False)

        updated: XrayConfig = remove_domain_from_config(xray_config, args.index, config, confirm=confirm)

        # Сохраняем и перезапускаем только если флаг -y был указан
        if confirm:
            write_remote_json(updated, config, create_backup=True)
            restart_service(config)
    except (ValueError, KeyError, TypeError) as e:
        get_logger().error(f'Ошибка при обработке команды remove: {e}')
        sys.exit(1)


def handle_restart_command(args: argparse.Namespace) -> None:
    """Handle restart command."""
    config = ConfigFile.from_file(args.config if hasattr(args, 'config') else None)
    restart_service(config)


def handle_backup_list_command(args: argparse.Namespace) -> None:
    """Handle backup list command."""
    config = ConfigFile.from_file(args.config if hasattr(args, 'config') else None)
    backup_manager = BackupManager(config.backup_dir, config.backup_count)
    backups = backup_manager.list_backups(config)
    if not backups:
        get_logger().info('Бэкапы не найдены')
        return

    for index, filename, timestamp in backups:
        get_logger().info(f'{index}. {filename} ({timestamp})')
    get_logger().info(f'\nВсего бэкапов: {len(backups)}')


def handle_backup_clean_command(args: argparse.Namespace) -> None:
    """Handle backup clean command."""
    config = ConfigFile.from_file(args.config if hasattr(args, 'config') else None)
    backup_manager = BackupManager(config.backup_dir, config.backup_count)
    backup_manager.cleanup_old_backups(config)
    get_logger().info('Старые бэкапы очищены')


def handle_backup_command(args: argparse.Namespace) -> None:
    """Handle backup command routing."""
    if args.backup_command == 'list':
        handle_backup_list_command(args)
    elif args.backup_command == 'clean':
        handle_backup_clean_command(args)
    elif args.backup_command == 'restore':
        handle_restore_command(args)
    else:
        get_logger().error('Неизвестная команда backup')
        sys.exit(1)


def handle_restore_command(args: argparse.Namespace) -> None:
    """Handle restore command."""
    if not args.preview and not args.confirm:
        get_logger().error('Укажите --preview для просмотра или --confirm для восстановления')
        sys.exit(1)

    config = ConfigFile.from_file(args.config if hasattr(args, 'config') else None)
    backup_manager = BackupManager(config.backup_dir, config.backup_count)

    if args.preview:
        content = backup_manager.restore_backup(args.index, config, preview_only=True)
        get_logger().info(f'Содержимое бэкапа #{args.index}:')
        if content:
            get_logger().info(content)

    if args.confirm:
        backup_manager.restore_backup(args.index, config, preview_only=False)
        get_logger().info(f'Бэкап #{args.index} восстановлен')
        restart_service(config)


def handle_status_command(args: argparse.Namespace) -> None:
    """Handle status command."""
    config = ConfigFile.from_file(args.config if hasattr(args, 'config') else None)
    if check_service_status_with_output(config):
        get_logger().info('Сервис работает корректно')
    else:
        get_logger().error('Сервис недоступен или есть ошибки конфигурации')


# ==== Command mapping ====
COMMANDS: dict[str, Callable[[argparse.Namespace], None]] = {
    'list': handle_list_command,
    'add': handle_add_command,
    'remove': handle_remove_command,
    'restart': handle_restart_command,
    'backup': handle_backup_command,
    'status': handle_status_command,
}


# ==== Main function ====
def main() -> None:
    """Main application entry point."""
    args: argparse.Namespace = parse_args()
    setup_logging(args.verbose if hasattr(args, 'verbose') else False)

    if not args.command:
        get_logger().error('Команда не указана. Используйте -h или --help для получения справки.')
        sys.exit(1)

    command_handler: Callable[[argparse.Namespace], None] | None = COMMANDS.get(args.command)
    if command_handler:
        command_handler(args)
    else:
        get_logger().error(f'Неизвестная команда: {args.command}')
        sys.exit(1)


if __name__ == '__main__':
    main()
