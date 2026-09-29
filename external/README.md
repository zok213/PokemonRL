# External Repositories Catalog

This directory contains shallow clones of the foundational open-source codebases referenced in the 2026 literature survey:

| Directory | Upstream Repository | Authors / Organization | Key Role / Reference |
|:---|:---|:---|:---|
| `pokered` | [pret/pokered](https://github.com/pret/pokered) | pret team | Canonical Game Boy LR35902 assembly disassembly, memory map, and symbol table |
| `PokemonRedExperiments` | [PWhiddy/PokemonRedExperiments](https://github.com/PWhiddy/PokemonRedExperiments) | Peter Whidden | Foundational PPO + pixel k-NN curiosity baseline (2023) |
| `pokemonred_puffer` | [drubinstein/pokemonred_puffer](https://github.com/drubinstein/pokemonred_puffer) | David Rubinstein, Joseph Suarez | PufferLib C-vectorized high-throughput environment (50k SPS) |
| `PokeRL` | [reddheeraj/PokemonRL](https://github.com/reddheeraj/PokemonRL) | Dheeraj Mudireddy, Sai Patibandla | Dynamic Action Masking, visited grid tracking & anti-looping (arXiv:2604.10812) |
| `metamon` | [UT-Austin-RPL/metamon](https://github.com/UT-Austin-RPL/metamon) | Jake Grigsby et al. | Showdown battle replay converter & offline AMAGO causal transformer (RLC 2025) |
| `continual-harness` | [sethkarten/continual-harness](https://github.com/sethkarten/continual-harness) | Seth Karten, Chi Jin et al. | NeurIPS PokéAgent Challenge competition & speedrunning track harness |

To re-clone or update all external repositories, run:
```bash
# Windows
.\scripts\clone_external_repos.bat

# Linux / macOS
bash scripts/clone_external_repos.sh
```
