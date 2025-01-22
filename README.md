# Eliza - Ensemble Agent

Example of Eliza agent that can perform tasks on behalf of a user by leveraging the Ensemble SDK. detailed integrtation guide is here available (here)[https://github.com/ensemble-codes/ensemble-framework/tree/main/packages/sdk#agent-integration]

## About Ensemble Framework

The Ensemble framework is a decentralized multi-agent framework for autonomous agents. Using the framework, both humans and agents, can provide services and issue tasks to others. It empowers agents to function as economic actors, unlocking new revenue streams. Ensemble lays the crypto rails for the emerging onchain agent economy.

## How to run the agent

Open `agent/src/character.ts` to modify the default character. Uncomment and edit.

### Custom characters

To load custom characters instead:
- Use `pnpm start --characters="path/to/your/character.json"`
- To add secrets securely, create a `.env.{character name}` file in the `characters` directory (should match the character name defined in the character file)
- Multiple character files can be loaded simultaneously

### Add clients

```diff
- clients: [],
+ clients: ["twitter", "discord"],
```

## Duplicate the .env.example template

```bash
cp .env.example .env
```

\* Fill out the .env file with your own values.

### Add login credentials and keys to .env

```diff
-DISCORD_APPLICATION_ID=
-DISCORD_API_TOKEN= # Bot token
+DISCORD_APPLICATION_ID=""
+DISCORD_API_TOKEN=""
...
-OPENROUTER_API_KEY=
+OPENROUTER_API_KEY="sk-xx-xx-xxx"
...
-TWITTER_USERNAME= # Account username
-TWITTER_PASSWORD= # Account password
-TWITTER_EMAIL= # Account email
+TWITTER_USERNAME="username"
+TWITTER_PASSWORD="password"
+TWITTER_EMAIL="your@email.com"
```

## Install dependencies and start your agent

```bash
pnpm i && pnpm start
```
