import wandb
import openai
from config_singleton import WandbConfigSingleton

def sample_evaluate():
    # Retrieve the instance from WandbConfigSingleton and load the W&B run and configuration
    instance = WandbConfigSingleton.get_instance()
    run = instance.run
    cfg = instance.config
    pipe = LLMPipeline.get()

    ## データセットを取得しましょう
    # 効率的な運用のために、少量データでのテストモードを別途作るようにしてください
    # 実装はSaaSだけでなく、dedicated cloudでも動くように、OpenAIだけでなく、Azure OpenAIでも動くように心がけてください
    #if cfg.testmode:
    #    artifact_dir = run.use_artifact(cfg.sample.dataset_test, type='dataset').download()
    #else:
    #    artifact_dir = run.use_artifact(cfg.sample.dataset, type='dataset').download()

    for ...
        output = pipe.generate()...
    
    some = wandb.Table(...)
    run.log({'some_able': some_table})
    

