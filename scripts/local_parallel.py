import curses
import subprocess
import threading
import itertools

from gin.config_parser import ConfigParser, BindingStatement


class GinParser:
    def __init__(self, file):
        self.parser = ConfigParser(file, None)
        lines = [line for line in self.parser]
        self.combinations = [self.get_combination_len(line) for line in lines]
        self.lines = [line if combs > 1 else line.location.line_content.strip() for line, combs in
                      zip(lines, self.combinations)]

    def __iter__(self):
        self._iter = itertools.product(*[range(x) for x in self.combinations])
        return self

    def __next__(self):
        ret = []
        for line, i in zip(self.lines, next(self._iter)):
            if isinstance(line, BindingStatement):
                ret.append('/'.join(
                    x for x in (line.scope, '.'.join(y for y in (line.selector, line.arg_name) if y != '')) if
                    x != '') + '=' + str(line.value[i]))
            else:
                ret.append(line)

        return '\n'.join(ret)

    @staticmethod
    def get_combination_len(line):
        if isinstance(line, BindingStatement) and isinstance(line.value,
                                                             list) and not line.location.line_content.strip().endswith(
                "skip"):
            return len(line.value)
        return 1


def gin_config_from_dict(params: dict):
    return '\n'.join(str(x) + "=" + str(y) for x, y in params.items())


def create_local_job(py_file, gin_params) -> subprocess.Popen:
    return subprocess.Popen(['/home/p/miniconda3/envs/workspace/bin/python', py_file], env={
        "GIN_CONFIG": gin_config_from_dict(gin_params) if isinstance(gin_params, dict) else str(gin_params)},
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


def parse_and_run(py_file, gin_file):
    with open(gin_file) as f:
        processes = [create_local_job(py_file, conf) for i, conf in enumerate(GinParser(f))]
    latest_logs = ["" for _ in processes]

    def stream_pod_logs(logs_list: [str], index: int):
        for line in iter(processes[index].stdout.readline, ''):
            logs_list[index] = line

    threads = [threading.Thread(target=stream_pod_logs, args=(latest_logs, i), daemon=True) for
               i, pod in enumerate(processes)]
    any(t.start() for t in threads)

    def print_logs_in_curses(stdscr):
        stdscr.nodelay(1)  # 0.1 second

        inp = -1
        while any(t.is_alive() for t in threads):
            stdscr.erase()
            if inp == -1:
                stdscr.addstr(0, 0, "Press e to exit")
            elif inp == ord('e'):
                break
            else:
                stdscr.addstr(0, 0, "unknown command")

            for i, log in enumerate(latest_logs):
                stdscr.addstr(i + 1, 0,
                              log)  # carriage returns mess up the output
                              # f'proc{i}' + ' | ' + log[log.rfind("\r") + 1:])  # carriage returns mess up the output
            stdscr.refresh()
            inp = stdscr.getch()

    curses.wrapper(print_logs_in_curses)


if __name__ == '__main__':
    parse_and_run('train.py', 'test.gin')
