"""
Multi-Attack Watermark Generation Script

Generates watermarks using multiple attack types for robust watermarking.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
from config import ConfigManager, ExperimentConfig
from adversarial_attacks import (
    generate_multiple_attacks,
    get_supported_attacks,
    get_attack_info
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description='Generate watermarks using multiple attack types',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Supported attack types:
{chr(10).join(f'  - {at}: {get_attack_info(at).get("name", "Unknown")}' for at in get_supported_attacks())}

Examples:
  # Generate with single attack (FGSM)
  python generate_watermarks_multi_attack.py --config configs/default.yaml --attack-types fgsm
  
  # Generate with multiple attacks
  python generate_watermarks_multi_attack.py --config configs/default.yaml --attack-types fgsm pgd bim
  
  # Generate with all supported attacks
  python generate_watermarks_multi_attack.py --config configs/default.yaml --attack-types all
        """
    )
    
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Path to YAML configuration file')
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to the trained model to attack')
    parser.add_argument('--attack-types', nargs='+', default=['fgsm'],
                       help='Attack types to use (default: fgsm). Use "all" for all supported attacks.')
    parser.add_argument('--eps', type=float, default=None,
                       help='Epsilon value (overrides config)')
    parser.add_argument('--sample-size', type=int, default=None,
                       help='Sample size (overrides config)')
    parser.add_argument('--output-dir', type=str, default='../data/adversarial',
                       help='Output directory for adversarial examples')
    
    args = parser.parse_args()
    
    # Load configuration
    config = None
    if os.path.exists(args.config):
        config = ConfigManager.load_from_yaml(args.config)
        watermark_config = config.watermark
        dataset_name = config.training.dataset_name
    else:
        logger.warning(f"Config file not found: {args.config}. Using defaults.")
        from config import WatermarkConfig, TrainingConfig
        watermark_config = WatermarkConfig()
        dataset_name = 'cifar10'  # Default dataset
    
    # Determine attack types
    if 'all' in args.attack_types:
        attack_types = get_supported_attacks()
        logger.info(f"Using all supported attacks: {attack_types}")
    else:
        attack_types = args.attack_types
    
    # Validate attack types
    supported = get_supported_attacks()
    invalid = [at for at in attack_types if at.lower() not in supported]
    if invalid:
        logger.error(f"Invalid attack types: {invalid}")
        logger.info(f"Supported types: {supported}")
        sys.exit(1)
    
    # Get parameters
    eps = args.eps if args.eps is not None else watermark_config.eps_values[0]
    sample_size = args.sample_size if args.sample_size is not None else watermark_config.sample_sizes[0]
    
    logger.info("=" * 80)
    logger.info("🚀 Multi-Attack Watermark Generation")
    logger.info("=" * 80)
    logger.info(f"Model: {args.model_path}")
    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"Attack types: {attack_types}")
    logger.info(f"Epsilon: {eps}")
    logger.info(f"Sample size: {sample_size}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info("")
    
    # Generate adversarial examples for each attack type
    results = generate_multiple_attacks(
        dataset_name=dataset_name,
        model_path=args.model_path,
        attack_types=attack_types,
        eps=eps,
        adversarial_sample_size=sample_size,
        clip_values=(0., 1.),
        attack_params=watermark_config.attack_params if hasattr(watermark_config, 'attack_params') else {},
        output_dir=args.output_dir
    )
    
    # Summary
    logger.info("")
    logger.info("=" * 80)
    logger.info("📊 Generation Summary")
    logger.info("=" * 80)
    for attack_type, (x_orig, x_adv, y_labels) in results.items():
        logger.info(f"✅ {attack_type.upper()}: Generated {len(x_adv)} adversarial examples")
    logger.info("")
    logger.info(f"All adversarial examples saved to: {args.output_dir}")
    logger.info("✅ Multi-attack watermark generation completed!")


if __name__ == '__main__':
    main()

