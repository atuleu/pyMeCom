import mecom
import glob
import time
import logging
import sys
import functools
import signal
import yaml
import matplotlib.pyplot as plt
import numpy as np

from collections import OrderedDict
import dataclasses


def _open_mecoms(pattern):
    res = {}

    for itf in glob.glob(pattern):
        dev = mecom.MeCom(itf)
        idt = dev.identify()
        res[(idt, itf)] = dev

    res = OrderedDict(sorted(res.items()))
    channels = {
        2 * i + 1: f"Bus:'{itf} Device:{idt} Instance:1'"
        for i, (idt, itf) in enumerate(res)
    }
    channels.update(
        {
            2 * i + 2: f"Bus:'{itf} Device:{idt} Instance:2'"
            for i, (idt, itf) in enumerate(res)
        }
    )
    channels = OrderedDict(sorted(channels.items()))

    logging.warning(f"Found {len(channels)} channels")
    for ch, loc in channels.items():
        logging.warning("Channel %d: is from %s", ch, loc)
    return res


@dataclasses.dataclass
class DeviceStatus:
    object_temperature: float = dataclasses.field(
        metadata={"name": "Object Temperature"}
    )
    sink_temperature: float = dataclasses.field(
        metadata={"name": "Sink Temperature"},
    )
    target_object_temperature: float = dataclasses.field(
        metadata={"name": "Target Object Temperature"}
    )
    output_current: float = dataclasses.field(
        metadata={"name": "Actual Output Current"}
    )
    output_voltage: float = dataclasses.field(
        metadata={"name": "Actual Output Voltage"}
    )
    temperature_is_stable: int = dataclasses.field(
        metadata={"name": "Temperature is Stable"}
    )
    status: int = dataclasses.field(metadata={"name": "Status"})

    stable_formats = {
        1: "\033[33m",
        2: "\033[36m",
    }

    @staticmethod
    def fetch_from_device(device, instance):
        fetch = functools.partial(device.get_parameter, parameter_instance=instance)
        kwargs = {
            f.name: fetch(f.metadata["name"]) for f in dataclasses.fields(DeviceStatus)
        }
        return DeviceStatus(**kwargs)

    @staticmethod
    def fetch_statuses(device):
        fields = dataclasses.fields(DeviceStatus)
        parameter_names = [f.metadata["name"] for f in fields]
        num_paremeters = len(parameter_names)
        parameter_names = [
            (p, instance) for instance in range(1, 3) for p in parameter_names
        ]
        data = device.get_bulk_parameters(parameter_names=parameter_names)
        instance1_status = DeviceStatus(
            **{f.name: d for f, d in zip(fields, data[:num_paremeters])}
        )
        instance2_status = DeviceStatus(
            **{f.name: d for f, d in zip(fields, data[num_paremeters:])}
        )
        return instance1_status, instance2_status

    def color_in(self):
        return DeviceStatus.stable_formats.get(self.temperature_is_stable, "")

    def color_out(self):
        if self.temperature_is_stable in DeviceStatus.stable_formats:
            return "\033[0m"
        return ""


def rounded_equal(a, b):
    return round(1000.0 * a) == round(1000.0 * b)


@dataclasses.dataclass
class AppConfig:
    window: float = 60.0
    loop: float = 0.5
    target: float = 15.0
    enabled: bool = False
    ramp: float = 5 / 60
    board_parameters: dict = dataclasses.field(default_factory=lambda: {})

    def status(self):
        if self.enabled:
            return 1
        return 0

    @staticmethod
    def load():
        kwargs = {}
        try:
            with open("config.yml") as f:
                kwargs = yaml.load(f, Loader=yaml.FullLoader)
                whitelist = [f.name for f in dataclasses.fields(AppConfig)]
                kwargs = {k: v for k, v in kwargs.items() if k in whitelist}
        except Exception as e:
            logging.warning("Could not read 'config.yml': %s", e)
        return AppConfig(**kwargs)


class App:
    def __init__(self, pattern="/dev/ttyUSB*"):
        file_handler = logging.FileHandler(filename="plate.log")
        file_handler.setLevel(logging.DEBUG)
        stream_handler = logging.StreamHandler(stream=sys.stderr)
        stream_handler.setLevel(logging.WARNING)
        logging.basicConfig(
            format="%(asctime)s %(message)s",
            handlers=[stream_handler, file_handler],
            level=logging.DEBUG,
        )
        logging.info("Starting")
        self._mecoms = _open_mecoms(pattern)

        self._config = AppConfig.load()
        self._target = float(self._config.target)
        self._last_target = float(self._target)
        self._set_time = 0.0

        self._initialize_mecoms()

        self._init_plot()

    @property
    def target(self):
        reach_time = abs(self._target - self._last_target) / self._config.ramp * 3600.0
        if reach_time == 0.0:
            return self._target
        now = time.time()
        ellapsed = max(now - self._set_time, 0)
        if ellapsed >= reach_time:
            return self._target
        return (
            self._last_target
            + (self._target - self._last_target) * ellapsed / reach_time
        )

    @target.setter
    def target(self, value):
        if value == self._target:
            return
        self._last_target = float(self.target)
        self._set_time = time.time()
        self._target = float(value)

    def _read_config(self):
        self._config = AppConfig.load()
        self.target = self._config.target

    def _initialize_mecoms(self):
        default_parameters = [
            {"name": "Save Data to Flash", "value": 1},
            {"name": "Status", "value": 0, "instance": 1},
            {"name": "Status", "value": 0, "instance": 2},
            {"name": "Sine Ramp Start Point", "value": 1, "instance": 1},
            {"name": "Sine Ramp Start Point", "value": 1, "instance": 2},
            {"name": "Kp", "value": 10.0, "instance": 1},
            {"name": "Kp", "value": 10.0, "instance": 2},
            {"name": "Ti", "value": 2.0, "instance": 1},
            {"name": "Ti", "value": 2.0, "instance": 2},
            {"name": "Td", "value": 300.0, "instance": 1},
            {"name": "Td", "value": 300.0, "instance": 2},
        ]

        for (idt, itf), m in self._mecoms.items():
            parameter_list = default_parameters + self._config.board_parameters.get(
                idt, []
            )
            for p in parameter_list:
                _update_parameters(
                    device=m,
                    identifiant=idt,
                    name=p["name"],
                    value=p.get("value", None),
                    instance=p.get("instance", 1),
                )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for m in self._mecoms.values():
            m.reset_device()
            m.stop()

    def loop(self):
        self._read_config()
        wanted_temperature = self.target
        statuses = self._send_fetch(wanted_temperature)
        self._update_plot(wanted_temperature, statuses)
        self._update_output(wanted_temperature, statuses)

    def _init_plot(self):
        self.fig, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        plt.ion()

        num_channels = 2 * len(self._mecoms)
        self.lines_a = [
            ax[0].plot([0], [0], linewidth=1)[0] for i in range(num_channels)
        ]
        self.lines_b = [
            ax[1].plot([0], [0], linewidth=1)[0] for i in range(num_channels)
        ]
        self.times = []
        self.temps = [[] for i in range(num_channels)]
        self.currents = [[] for i in range(num_channels)]
        ax[0].set_ylabel("Temperature (°C)")
        ax[0].legend(
            [f"CH {i+1}" for i in range(num_channels)],
            loc="upper center",
            bbox_to_anchor=[0.5, 1.3],
            ncol=5,
        )
        ax[0].grid()
        ax[0].set_xlim([-self._config.window, 0])
        ax[0].set_ylim([-2, 25.0])

        ax[1].grid()
        ax[1].set_ylabel("Current (A)")
        ax[1].set_xlabel("Time (min)")
        ax[1].set_ylim([-6, 6])
        plt.show()

    def _update_plot(self, target, statuses):
        self.times.append(time.time())
        new_start = 0
        while (self.times[-1] - self.times[new_start]) > (self._config.window * 60.0):
            new_start += 1
        for temps, currents, s in zip(self.temps, self.currents, statuses):
            temps.append(s.object_temperature)
            currents.append(s.output_current)
        for i in range(len(self.temps)):
            self.temps[i] = self.temps[i][new_start:]
            self.currents[i] = self.currents[i][new_start:]
        self.times = self.times[new_start:]

        times = np.array(self.times)
        times -= self.times[-1]
        times /= 60.0
        for temp, line in zip(self.temps, self.lines_a):
            line.set_xdata(times)
            line.set_ydata(temp)

        for current, line in zip(self.currents, self.lines_b):
            line.set_xdata(times)
            line.set_ydata(current)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def _update_output(self, target, statuses):
        print()
        print(
            "Target: %6.3f °C, Interpolated: %6.3f °C Enabled %-5s"
            % (self._config.target, target, self._config.enabled)
        )
        print("┌─────────┬─────────────┬───────────┬─────────────┬─────────────┐")
        print("│ Channel │ Object (°C) │ Sink (°C) | Current (A) │ Tension (V) │")
        print("├─────────┼─────────────┼───────────┼─────────────┼─────────────┤")
        for i, s in enumerate(statuses):
            print(
                "│%s %7d │ %-11.3f │ %-9.3f │ %-11.3f │ %-11.3f %s│"
                % (
                    s.color_in(),
                    i + 1,
                    s.object_temperature,
                    s.sink_temperature,
                    s.output_current,
                    s.output_voltage,
                    s.color_out(),
                )
            )
        print("└─────────┴─────────────┴───────────┴─────────────┴─────────────┘")
        print(f"\033[{len(statuses)+7}A")

    def _send_fetch(self, target):
        res = []
        for m in self._mecoms.values():
            s1, s2 = DeviceStatus.fetch_statuses(m)
            self._update_state(s1, m, 1, target)
            self._update_state(s2, m, 2, target)
            res.extend([s1, s2])

        return res

    def _update_state(self, status, device, instance, target):
        if rounded_equal(status.target_object_temperature, target) is False:
            device.set_parameter(
                parameter_name="Target Object Temp (Set)",
                value=target,
                parameter_instance=instance,
            )

        if status.status != self._config.status():
            device.set_parameter(
                parameter_name="Status",
                value=self._config.status(),
                parameter_instance=instance,
            )


def _update_parameters(*, device, identifiant, name, value=None, instance=1):
    if value is None:
        logging.warning(
            "Device %d, parameter:'%s', instance:%d, current_value:%s",
            identifiant,
            name,
            instance,
            device.get_parameter(
                parameter_name=name,
                parameter_instance=instance,
            ),
        )
    else:
        device.set_parameter(
            parameter_name=name, value=value, parameter_instance=instance
        )


should_continue = True


def signal_handler(signal, frame):
    global should_continue
    logging.info("Received SIGINT")
    should_continue = False


signal.signal(signal.SIGINT, signal_handler)


def main():
    with App() as app:
        next_iteration = time.time() + app._config.loop
        while should_continue:
            app.loop()
            now = time.time()
            to_wait = next_iteration - now
            next_iteration += app._config.loop
            if to_wait > 0.0:
                time.sleep(to_wait)


if __name__ == "__main__":
    main()
