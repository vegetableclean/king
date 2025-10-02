import argparse
import glob
import json
import re
import numpy as np


class ResultsParser():
    def __init__(self, args):
        """
        """
        self.args = args

        self.results_files = self.parse_results_dir()
        print(f"Found {len(self.results_files)} results files...")

        with open('./tools/timings.json') as f:
            self.timings = json.load(f)

    def parse_results_dir(self):
        """
            Parse the results directory and gather the
            relevant JSON file paths from
            the "RouteScenario_*_to_*" subdirectories.
        """
        route_scenario_dirs = sorted(
            glob.glob(
                self.args.results_dir + "/**/RouteScenario_*", recursive=True
            ),
            key=lambda path: int(path.split("_")[-1]),
        )

        results_files = []
        for dir in route_scenario_dirs:
            results_files.extend(
                sorted(
                    glob.glob(dir + "/results.json")
                )
            )

        return results_files

    def parse_json_file(self, records_file):
        """
        """
        return json.loads(open(records_file).read())

    def generate_report(self):
        """
        """
        iterations = []
        num_routes = len(self.results_files)
        optim_method = str(self.args.optim_method)

        # normalize optimizer names to match timings.json
        om_map = {
            'adam': 'Adam',
            'both_paths': 'Both_Paths',
            'both-paths': 'Both_Paths',
            'bothpaths': 'Both_Paths',
        }
        optim_method_norm = om_map.get(optim_method.lower(), optim_method)

        def norm_density(tok: str) -> str:
            """normalize traffic density labels"""
            if tok is None:
                return str(self.args.num_agents)
            t = str(tok).strip().lower()
            td_map = {
                'e': 'easy', 'easy': 'easy',
                'm': 'medium', 'med': 'medium',
                'h': 'hard', 'hard': 'hard'
            }
            if t.isdigit():
                return t
            return td_map.get(t, t)

        for results_file in self.results_files:
            results = self.parse_json_file(results_file)

            # try to detect traffic density from path
            traffic_density_raw = None
            for part in results_file.split('/'):
                m = re.search(r'agents_([A-Za-z0-9\-]+)', part)
                if m:
                    traffic_density_raw = m.group(1)
                    break
            if traffic_density_raw is None:
                traffic_density_raw = str(self.args.num_agents)

            traffic_density = norm_density(traffic_density_raw)
            town = str(results.get("meta_data", {}).get("town", "")).strip()

            # compute factor with safe fallback
            if self.args.use_GPU_hours:
                factor = (
                    self.timings.get(optim_method_norm, {})
                                .get(traffic_density, {})
                                .get(town)
                )
                if factor is None:
                    print(f"[WARN] Missing timings for method={optim_method_norm}, "
                          f"density={traffic_density}, town={town}. Using factor=1.0")
                    factor = 1.0
            else:
                factor = 1.0

            fm = results.get("first_metrics", {})
            collided = (fm.get("Collision Metric") == 1)
            iters = fm.get("iteration", 0)

            if collided and iters * factor < float(self.args.max_GPU_hours) * 60 * 60:
                iterations.append(iters * factor)

        iterations.sort()

        print(f'Collision rate: {len(iterations)/num_routes}')
        if int(50*num_routes/100) < len(iterations):
            print(f't@50: {np.mean(iterations[:int(50*num_routes/100)])}')
        else:
            print('CR is lower than 50%, not computing t@50.')
        print('-------------------------------------------')


if __name__ == '__main__':
    main_parser = argparse.ArgumentParser()
    main_parser.add_argument(
        "--results_dir",
        type=str,
        default="./outputs/",
        help="The directory containing the scenario records and results files \
              for each of the optimized scenarios/routes for each traffic density.",
    )
    main_parser.add_argument(
        "--use_GPU_hours",
        type=int,
        default=1,
    )
    main_parser.add_argument(
        "--max_GPU_hours",
        type=float,
        default=0.05,
    )
    main_parser.add_argument(
        "--optim_method",
        default="Adam",
        choices=["Adam", "Both_Paths"]
    )
    main_parser.add_argument(
        "--num_agents",
        type=int,
        default=4
    )

    args = main_parser.parse_args()

    results_parser = ResultsParser(args)
    results_parser.generate_report()

