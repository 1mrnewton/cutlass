<p align="center">
  <img src="../../assets/app-icon.PNG" alt="Cutlass" width="112">
</p>

<h1 align="center">Cutlass</h1>

<p align="center">
  <b>Um editor de vídeo gratuito e de código aberto, com um assistente com quem você pode conversar.</b><br>
  Para macOS, Windows e Linux.
</p>

<p align="center">
  <a href="https://github.com/1mrnewton/cutlass/releases/latest"><img alt="Latest release" src="https://img.shields.io/github/v/release/1mrnewton/cutlass?style=flat-square&label=release"></a>
  <a href="https://github.com/1mrnewton/cutlass/actions/workflows/ci.yml"><img alt="Build status" src="https://img.shields.io/github/actions/workflow/status/1mrnewton/cutlass/ci.yml?branch=main&style=flat-square&label=build"></a>
  <a href="#license"><img alt="License: MIT or Apache-2.0" src="https://img.shields.io/badge/license-MIT%20%7C%20Apache--2.0-blue?style=flat-square"></a>
</p>

<p align="center">
  <a href="#download">Baixar</a> ·
  <a href="#what-you-can-do">Recursos</a> ·
  <a href="#the-assistant">Assistente</a> ·
  <a href="#formats">Formatos</a> ·
  <a href="#build-from-source">Compilar</a> ·
  <a href="#not-here-yet">O que falta</a> ·
  <a href="https://cutlass.sh">Site</a>
</p>

[English](../../README.md) · [简体中文](README.zh-CN.md) · [繁體中文](README.zh-TW.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Español](README.es.md) · **Português** · [Français](README.fr.md) · [Deutsch](README.de.md) · [Italiano](README.it.md) · [Русский](README.ru.md) · [العربية](README.ar.md) · [हिन्दी](README.hi.md) · [Bahasa Indonesia](README.id.md)

O Cutlass é um editor de vídeo com linha do tempo. Importe vídeo, som e imagens, corte tudo em trilhas, adicione texto, efeitos e cor, e exporte um vídeo pronto. Você pode fazer cada etapa à mão, ou descrever a edição em linguagem simples e deixar o assistente fazê-la na linha do tempo, onde você pode alterá-la ou desfazê-la.

É gratuito. Não há conta, marca d'água nem envio de arquivos. Seu material fica no seu disco, e as ferramentas que usam aprendizado de máquina rodam no seu próprio computador.

Por baixo do editor, os codecs são nossos. H.264, HEVC, AV1, ProRes, AAC, Opus e os demais são escritos em Rust neste repositório. O Cutlass não inclui o FFmpeg.

<a name="download"></a>
## Baixar

| Sistema | Arquivo |
| --- | --- |
| macOS, Apple silicon | [Cutlass.dmg](https://github.com/1mrnewton/cutlass/releases/latest/download/Cutlass.dmg) |
| Windows, 64 bits | [Cutlass-setup.exe](https://github.com/1mrnewton/cutlass/releases/latest/download/Cutlass-setup.exe) |
| Linux, x86_64 | [Cutlass-x86_64.AppImage](https://github.com/1mrnewton/cutlass/releases/latest/download/Cutlass-x86_64.AppImage) |

Cada arquivo tem ao lado uma soma de verificação `.sha256` na página de [versões](https://github.com/1mrnewton/cutlass/releases). Depois de instalado, o Cutlass procura sozinho por uma versão mais nova: **Check for updates…** no menu.

As compilações ainda não são assinadas com um certificado de desenvolvedor, então a primeira abertura tem um passo a mais:

- **macOS:** abra o app uma vez e depois permita em System Settings › Privacy & Security › **Open Anyway**.
- **Windows:** se o SmartScreen barrar o instalador, escolha **More info** › **Run anyway**.
- **Linux:** torne o arquivo executável com `chmod +x Cutlass-x86_64.AppImage` e depois execute-o.

<a name="what-you-can-do"></a>
## O que você pode fazer

**Corte em uma linha do tempo.** Trilhas de vídeo, áudio e texto. Divida, apare, feche lacunas, congele um quadro, inverta, espelhe, gire e recorte. Agrupe clipes, ou dobre-os em um clipe composto e abra-o depois. Substitua um clipe mantendo suas edições, ou copie os atributos de um clipe para outro. Material pesado pode ser cortado por meio de um proxy.

**Velocidade e movimento.** Defina uma velocidade constante, ou escolha uma curva de velocidade e mude sua forma arrastando os pontos. Anime um clipe com quadros-chave, e mova ou apague vários de uma vez. Animações de entrada, de saída e combinadas estão prontas para aplicar.

**Cor.** Ajustes básicos, HSL, curvas e rodas de cor. Carregue uma LUT `.cube` e mantenha os tons de pele naturais sob ela. 200 filtros, e camadas de ajuste na linha do tempo.

**Efeitos e transições.** 200 efeitos, 180 transições e 13 modos de mesclagem. Mascare um clipe com uma linha, um espelho, um círculo, um retângulo, um coração ou uma estrela. Remova um fundo verde, ou clique em um objeto para recortá-lo.

**Texto, legendas e figurinhas.** Texto com fontes, contorno, sombra, fundo, balões e efeitos, animado na entrada, na saída ou em loop. As legendas são ouvidas do clipe na sua máquina (por enquanto em inglês) e depois editadas como uma lista: redigite uma linha, localize e substitua, escolha um modelo, estilize palavras isoladas e exporte o resultado como arquivo SRT. Figurinhas e formas ficam na linha do tempo como qualquer outro clipe.

**Ferramentas de imagem que rodam no seu computador.** Estabilizar, melhorar a qualidade, reduzir o ruído da imagem, fluxo óptico para uma câmera lenta suave, reiluminar, reenquadramento automático e desfoque de movimento. Nada é enviado a lugar nenhum.

**Som.** Volume e fades, extrair o áudio de um vídeo, sincronizar vídeo e áudio pelo som, e marcar a batida.

**Exportar.** MP4 ou MOV: a linha do tempo inteira, um intervalo ou os clipes selecionados. Também só o som, um GIF ou um WebP animado. No macOS e no Windows, os codecs de hardware do sistema fazem o trabalho quando podem, e uma chave nos ajustes os desliga.

<a name="the-assistant"></a>
## O assistente

Diga o que você quer, e o assistente edita o projeto aberto com as mesmas requisições que a linha do tempo envia. Cada alteração que ele faz é um único passo de desfazer.

- **Ele roda na sua máquina.** O assistente conversa com um executor de modelos no seu próprio computador, como o [Ollama](https://ollama.com), o servidor do llama.cpp ou o LM Studio. Defina o endereço e o modelo em Settings › Assistant. Por esse caminho, o prompt não sai do seu computador.
- **Ele pergunta antes.** Tudo o que enviaria dados para fora da máquina espera você permitir.
- **Ele tem habilidades.** Legendas, remoção de pausas, motion graphics, uma demonstração de produto e a configuração do projeto vêm com o app, e ele pode chamar ferramentas dos servidores MCP que você adicionar.
- **Outros agentes também podem controlar o Cutlass.** `cutlass mcp /path/to/project` oferece as mesmas ferramentas por stdio, sem janela, então o Claude ou qualquer cliente MCP pode ler e editar um projeto.

<a name="formats"></a>
## Formatos

Os decodificadores e codificadores abaixo são código Rust em [`crates/codecs`](../../crates/codecs) e [`crates/containers`](../../crates/containers), testado contra o FFmpeg como referência externa. O README de cada crate diz o que ele suporta e como sua velocidade se compara à do FFmpeg.

| | Abrir | Exportar |
| --- | :---: | :---: |
| **Arquivos** | MP4, MOV, M4V, M4A, WAV, AIFF, FLAC, Ogg, MP3 | MP4, MOV, e só o som |
| **Vídeo** | H.264, HEVC, AV1, ProRes, VP8, MPEG-4, MPEG-2, Motion JPEG | H.264, ProRes, VP8, MPEG-4, MPEG-2, Motion JPEG. HEVC pelo codificador do sistema no macOS e no Windows |
| **Som** | AAC, MP3, FLAC, ALAC, Opus, Vorbis, AC-3, E-AC-3, PCM | AAC, MP3, FLAC, ALAC, Opus, Vorbis, AC-3, E-AC-3, PCM |
| **Imagens** | JPEG, PNG, WebP, GIF, animados ou estáticos | GIF, WebP animado |

<a name="build-from-source"></a>
## Compilar a partir do código-fonte

Você precisa da cadeia de ferramentas estável do [Rust](https://rustup.rs) e de um compilador C. No Linux, também dos cabeçalhos do GTK e do ALSA:

```sh
sudo apt-get install libgtk-3-dev libasound2-dev
```

Depois:

```sh
git clone https://github.com/1mrnewton/cutlass
cd cutlass
cargo run --release -p cutlass
```

As verificações que a CI executa (a última precisa do [cargo-nextest](https://nexte.st)):

```sh
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo nextest run --workspace
```

<a name="not-here-yet"></a>
## O que falta

O Cutlass é jovem. No app, um recurso que está desenhado mas ainda não foi construído aparece esmaecido e marcado como **Later**, então você sempre sabe o que funciona. As maiores lacunas hoje:

- **Mais contêineres.** Arquivos MKV, WebM, AVI e MPEG-TS ainda não abrem.
- **Ferramentas de voz.** Redução de ruído, realce de voz, normalização de volume, isolamento de voz e texto para fala.
- **Ferramentas generativas.** Remover, expandir e estilizar com IA.
- **Rastreamento.** Texto, figurinhas e máscaras que seguem um objeto.
- **Legendas em mais idiomas.**
- **Um modelo hospedado para o assistente.** Hoje ele usa um modelo local.
- **Celulares.** Os apps de iOS e Android estão em [`apps`](../../apps) e compartilham o mesmo núcleo em Rust, mas o desktop vem primeiro.

Um relato de bug com o arquivo que o causou é a coisa mais útil que você pode enviar. [Abra uma issue](https://github.com/1mrnewton/cutlass/issues).

## Como o código está organizado

| Pasta | O que há nela |
| --- | --- |
| [`apps/desktop`](../../apps/desktop) | O editor para desktop, uma janela egui. Compila o binário `cutlass` |
| [`apps/ios`](../../apps/ios), [`apps/android`](../../apps/android) | Os apps para celular, em SwiftUI e Kotlin Compose. Eles acessam o núcleo em Rust por meio de `crates/ffi` |
| [`crates/model`](../../crates/model), [`crates/session`](../../crates/session), [`crates/editor`](../../crates/editor) | O projeto, as requisições de edição com desfazer, e o comportamento do editor compartilhado por todos os apps |
| [`crates/eval`](../../crates/eval), [`crates/engine`](../../crates/engine), [`crates/render`](../../crates/render), [`crates/playback`](../../crates/playback), [`crates/export`](../../crates/export) | Transformar um projeto e um instante em um quadro e seu som, para a pré-visualização e para a exportação |
| [`crates/codecs`](../../crates/codecs), [`crates/containers`](../../crates/containers) | Um crate por codec e por formato de arquivo |
| [`crates/platform-apple`](../../crates/platform-apple), [`crates/platform-windows`](../../crates/platform-windows) | Os codecs de hardware do sistema |
| [`crates/effects`](../../crates/effects), [`crates/text`](../../crates/text), [`crates/captions`](../../crates/captions) | Filtros, efeitos, transições, máscaras, cor, texto e legendas |
| [`crates/vision`](../../crates/vision), [`crates/transcription`](../../crates/transcription) | Os modelos de imagem e de fala que rodam no dispositivo |
| [`crates/agent`](../../crates/agent) | O assistente: suas ferramentas, suas habilidades e o MCP |

## Contribuir

Use o Cutlass com material real e nos diga onde ele falha. Os [formulários de issue](https://github.com/1mrnewton/cutlass/issues/new/choose) cobrem bugs, recursos, desempenho e arquivos que não abrem.

Vai enviar código? Execute antes as três verificações acima. Os crates compartilhados não dependem de nenhuma plataforma específica, então a mesma edição funciona no desktop e no celular.

<a name="license"></a>
## Licença

O Cutlass é gratuito sob qualquer uma das duas licenças. Escolha uma.

* [Apache License 2.0](../../LICENSE-APACHE)
* [MIT License](../../LICENSE-MIT)

Copyright (c) 2026 1mrnewton

