This code is based on the work from the [Android World](https://github.com/google-research/android_world/tree/main) team. We extend our sincere gratitude to them for their foundational research.

Our `vimo_empower` agent, `t3a_vimo`, can be found in the `./android_world/agents` directory.

### Running Experiments

To run the experiments, use the following command:

```bash
python run_wm_api.py --suite_family=android_world --agent_name=t3a_vimo_gemini --checkpoint_dir="<path_to_save_results>" --vimo_api="<your_vimo_api_address>"

**Argument Descriptions:**

* `--checkpoint_dir`: The file path where results will be saved.
* `--vimo_api`: The network address for your running VIMO API instance.
