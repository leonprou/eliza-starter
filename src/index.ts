import { PostgresDatabaseAdapter } from "@ai16z/adapter-postgres";
import { SqliteDatabaseAdapter } from "@ai16z/adapter-sqlite";
import { AutoClientInterface } from "@ai16z/client-auto";
import { DiscordClientInterface } from "@ai16z/client-discord";
import { TelegramClientInterface } from "@ai16z/client-telegram";
import { TwitterClientInterface } from "@ai16z/client-twitter";
import { Ensemble } from "@ensemble-ai/sdk";

import {
    AgentRuntime,
    CacheManager,
    Character,
    Clients,
    DbCacheAdapter,
    defaultCharacter,
    elizaLogger,
    FsCacheAdapter,
    IAgentRuntime,
    ICacheManager,
    IDatabaseAdapter,
    IDatabaseCacheAdapter,
    ModelProviderName,
    settings,
    stringToUuid,
    validateCharacterConfig,
} from "@ai16z/eliza";
import { bootstrapPlugin } from "@ai16z/plugin-bootstrap";
import { DirectClient } from "@ai16z/client-direct";
import { createNodePlugin } from "@ai16z/plugin-node";
import { solanaPlugin } from "@ai16z/plugin-solana";
import Database from "better-sqlite3";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import yargs from "yargs";
import dotenv from "dotenv";
import { ethers } from 'ethers';

// Creating a signer with a private key
export const createSigner = () => {
  const provider = new ethers.JsonRpcProvider(process.env.NETWORK_RPC_URL!, undefined, { polling: true});
  const pk = process.env.PRIVATE_KEY!;
  const wallet = new ethers.Wallet(pk, provider);

  return {
    provider,
    signer: wallet
  };
}

// Creating the signer
const { signer } = createSigner();

// Ensemble Config
const config = {
  taskRegistryAddress: process.env.TASK_REGISTRY_ADDRESS,
  agentRegistryAddress: process.env.AGENT_REGISTRY_ADDRESS,
  serviceRegistryAddress: process.env.SERVICE_REGISTRY_ADDRESS,
  network: {
    chainId: parseInt(process.env.NETWORK_CHAIN_ID),
    name: process.env.NETWORK_NAME,
    rpcUrl: process.env.NETWORK_RPC_URL,
  },
}

// validating the config
console.log({config})

// creating the ensemble sdk
const ensemble = new Ensemble(config, signer);

// starting the sdk listener
ensemble.start();

const __filename = fileURLToPath(import.meta.url); // get the resolved path to the file
const __dirname = path.dirname(__filename); // get the name of the directory

export const wait = (minTime: number = 1000, maxTime: number = 3000) => {
  const waitTime =
    Math.floor(Math.random() * (maxTime - minTime + 1)) + minTime;
  return new Promise((resolve) => setTimeout(resolve, waitTime));
};

export function parseArguments(): {
  character?: string;
  characters?: string;
} {
  try {
    return yargs(process.argv.slice(2))
      .option("character", {
        type: "string",
        description: "Path to the character JSON file",
      })
      .option("characters", {
        type: "string",
        description: "Comma separated list of paths to character JSON files",
      })
      .parseSync();
  } catch (error) {
    console.error("Error parsing arguments:", error);
    return {};
  }
}

export async function loadCharacters(
  charactersArg: string
): Promise<Character[]> {
  let characterPaths = charactersArg?.split(",").map((filePath) => {
    if (path.basename(filePath) === filePath) {
      filePath = "../characters/" + filePath;
    }
    return path.resolve(process.cwd(), filePath.trim());
  });

  const loadedCharacters = [];

  if (characterPaths?.length > 0) {
    for (const characterPath of characterPaths) {
      try {
        // Load character JSON
        const character = JSON.parse(fs.readFileSync(characterPath, "utf8"));
        validateCharacterConfig(character);

        // Load corresponding .env file
        const characterName = path.basename(characterPath, '.character.json');
        const envPath = path.join(path.dirname(characterPath), `.env.${characterName}`);
        
        if (fs.existsSync(envPath)) {
          const rawEnv = fs.readFileSync(envPath, 'utf8');
          const envConfig = dotenv.parse(rawEnv);
          character.settings = character.settings || {};
          character.settings.secrets = envConfig;
        }

        loadedCharacters.push(character);
      } catch (e) {
        console.error(`Error loading character from ${characterPath}: ${e}`);
        process.exit(1);
      }
    }
  }

  if (loadedCharacters.length === 0) {
    console.log("No characters found, using default character");
    loadedCharacters.push(defaultCharacter);
  }

  return loadedCharacters;
}

export function getTokenForProvider(
  provider: ModelProviderName,
  character: Character
) {
  switch (provider) {
    case ModelProviderName.OPENAI:
      return (
        character.settings?.secrets?.OPENAI_API_KEY || settings.OPENAI_API_KEY
      );
    case ModelProviderName.LLAMACLOUD:
      return (
        character.settings?.secrets?.LLAMACLOUD_API_KEY ||
        settings.LLAMACLOUD_API_KEY ||
        character.settings?.secrets?.TOGETHER_API_KEY ||
        settings.TOGETHER_API_KEY ||
        character.settings?.secrets?.XAI_API_KEY ||
        settings.XAI_API_KEY ||
        character.settings?.secrets?.OPENAI_API_KEY ||
        settings.OPENAI_API_KEY
      );
    case ModelProviderName.ANTHROPIC:
      return (
        character.settings?.secrets?.ANTHROPIC_API_KEY ||
        character.settings?.secrets?.CLAUDE_API_KEY ||
        settings.ANTHROPIC_API_KEY ||
        settings.CLAUDE_API_KEY
      );
    case ModelProviderName.REDPILL:
      return (
        character.settings?.secrets?.REDPILL_API_KEY || settings.REDPILL_API_KEY
      );
    case ModelProviderName.OPENROUTER:
      return (
        character.settings?.secrets?.OPENROUTER || settings.OPENROUTER_API_KEY
      );
    case ModelProviderName.GROK:
      return character.settings?.secrets?.GROK_API_KEY || settings.GROK_API_KEY;
    case ModelProviderName.HEURIST:
      return (
        character.settings?.secrets?.HEURIST_API_KEY || settings.HEURIST_API_KEY
      );
    case ModelProviderName.GROQ:
      return character.settings?.secrets?.GROQ_API_KEY || settings.GROQ_API_KEY;
  }
}

class DatabaseConnectionWrapper {
  private db: IDatabaseAdapter & IDatabaseCacheAdapter;
  private dataDir: string;
  private retryCount: number = 0;
  private maxRetries: number = 5;
  private retryDelay: number = 5000; // 5 seconds
  private keepAliveInterval: NodeJS.Timeout | null = null;
  private _isHealthy: boolean = false;

  constructor(dataDir: string) {
    this.dataDir = dataDir;
    this.db = this.initializeDatabase();
    this.startKeepAlive();
  }

  get isHealthy(): boolean {
    return this._isHealthy;
  }

  private initializeDatabase(): IDatabaseAdapter & IDatabaseCacheAdapter {
    if (process.env.POSTGRES_URL) {
      return new PostgresDatabaseAdapter({
        connectionString: process.env.POSTGRES_URL,
        pool: {
          max: 20,
          min: 5,
          idleTimeoutMillis: 300000,
          connectionTimeoutMillis: 10000,
          allowExitOnIdle: false,
          keepAlive: true,
          keepAliveInitialDelayMillis: 10000
        }
      }) as IDatabaseAdapter & IDatabaseCacheAdapter;
    } else {
      const filePath = process.env.SQLITE_FILE ?? path.resolve(this.dataDir, "db.sqlite");
      return new SqliteDatabaseAdapter(new Database(filePath)) as IDatabaseAdapter & IDatabaseCacheAdapter;
    }
  }

  private startKeepAlive() {
    // Run a keep-alive query every minute
    this.keepAliveInterval = setInterval(async () => {
      try {
        await (this.db as any).query('SELECT 1');
        elizaLogger.log('Keep-alive query successful');
        this._isHealthy = true;
      } catch (error) {
        elizaLogger.error('Keep-alive query failed:', error);
        this._isHealthy = false;
        await this.reconnect();
      }
    }, 60000);
  }

  private async reconnect() {
    elizaLogger.log('Attempting to reconnect to database...');
    this.db = this.initializeDatabase();
    await this.init();
  }

  async init() {
    while (this.retryCount < this.maxRetries) {
      try {
        await (this.db as IDatabaseAdapter).init();
        elizaLogger.log("Database connection established successfully");
        this.retryCount = 0; // Reset retry count on successful connection
        this._isHealthy = true;
        return;
      } catch (error) {
        this.retryCount++;
        elizaLogger.error(`Database initialization error (attempt ${this.retryCount}/${this.maxRetries}):`, error);
        this._isHealthy = false;
        
        if (this.retryCount < this.maxRetries) {
          elizaLogger.log(`Retrying database connection in ${this.retryDelay/1000} seconds...`);
          await new Promise(resolve => setTimeout(resolve, this.retryDelay));
          this.db = this.initializeDatabase();
        } else {
          throw new Error(`Failed to initialize database after ${this.maxRetries} attempts`);
        }
      }
    }
  }

  async cleanup() {
    if (this.keepAliveInterval) {
      clearInterval(this.keepAliveInterval);
    }
    try {
      await (this.db as any).end?.();
    } catch (error) {
      elizaLogger.error('Error cleaning up database connection:', error);
    }
  }

  getAdapter(): IDatabaseAdapter & IDatabaseCacheAdapter {
    return this.db;
  }
}

function isFalsish(input: any): boolean {
  // If the input is exactly NaN, return true
  if (Number.isNaN(input)) {
      return true;
  }

  // Convert input to a string if it's not null or undefined
  const value = input == null ? "" : String(input);

  // List of common falsish string representations
  const falsishValues = [
      "false",
      "0",
      "no",
      "n",
      "off",
      "null",
      "undefined",
      "",
  ];

  // Check if the value (trimmed and lowercased) is in the falsish list
  return falsishValues.includes(value.trim().toLowerCase());
}

function getSecret(character: Character, secret: string) {
  return character.settings?.secrets?.[secret] || process.env[secret];
}

export async function initializeClients(
  character: Character,
  runtime: IAgentRuntime
) {
  // each client can only register once
  // and if we want two we can explicitly support it
  const clients: Record<string, any> = {};
  const clientTypes: string[] =
      character.clients?.map((str) => str.toLowerCase()) || [];
  elizaLogger.log("initializeClients", clientTypes, "for", character.name);

  if (clientTypes.includes(Clients.DIRECT)) {
      const autoClient = await AutoClientInterface.start(runtime);
      if (autoClient) clients.auto = autoClient;
  }

  if (clientTypes.includes(Clients.DISCORD)) {
      const discordClient = await DiscordClientInterface.start(runtime);
      if (discordClient) clients.discord = discordClient;
  }

  if (clientTypes.includes(Clients.TELEGRAM)) {
      const telegramClient = await TelegramClientInterface.start(runtime);
      if (telegramClient) clients.telegram = telegramClient;
  }

  if (clientTypes.includes(Clients.TWITTER)) {
    console.log('starting the twitter client')
    const twitterClient = await TwitterClientInterface.start(runtime);

    if (twitterClient) {
        clients.twitter = twitterClient;
        (twitterClient as any).enableSearch = !isFalsish(
            getSecret(character, "TWITTER_SEARCH_ENABLE")
        );
    }
}

  if (clientTypes.includes(Clients.TWITTER)) {
    // const agentHubClient = await AgentHubClientInterface.start(runtime);

    // if (agentHubClient) {
    //     clients.twitterClient = agentHubClient;
    //     // (agentHubClient as any).enableSearch = !isFalsish(
    //     //     getSecret(character, "TWITTER_SEARCH_ENABLE")
    //     // );
    // }
  }

  elizaLogger.log("client keys", Object.keys(clients));


  if (character.plugins?.length > 0) {
      for (const plugin of character.plugins) {
          if (plugin.clients) {
              for (const client of plugin.clients) {
                  clients.push(await client.start(runtime));
              }
          }
      }
  }

  return clients;
}


export function createAgent(
  character: Character,
  db: IDatabaseAdapter,
  cache: ICacheManager,
  token: string
) {
  const nodePlugin = createNodePlugin();

  elizaLogger.success(
    elizaLogger.successesTitle,
    "Creating runtime for character",
    character.name
  );
  return new AgentRuntime({
    databaseAdapter: db,
    token,
    modelProvider: character.modelProvider,
    evaluators: [],
    character,
    plugins: [
      bootstrapPlugin,
      nodePlugin,
      character.settings.secrets?.WALLET_PUBLIC_KEY ? solanaPlugin : null,
    ].filter(Boolean),
    providers: [],
    actions: [],
    services: [],
    managers: [],
    cacheManager: cache,
  });
}

function intializeDbCache(character: Character, db: IDatabaseCacheAdapter) {
  const cache = new CacheManager(new DbCacheAdapter(db, character.id));
  return cache;
}


// async function handleTask(taskId: string) {
//   console.log('taskId', taskId);
// }

async function startAgent(character: Character, directClient: DirectClient) {
  let dbWrapper: DatabaseConnectionWrapper | null = null;
  let runtime: AgentRuntime | null = null;
  let clients: any[] = [];

  try {
    character.id ??= stringToUuid(character.name);

    const token = getTokenForProvider(character.modelProvider, character);
    const dataDir = path.join(__dirname, "../data");

    if (!fs.existsSync(dataDir)) {
      fs.mkdirSync(dataDir, { recursive: true });
    }

    dbWrapper = new DatabaseConnectionWrapper(dataDir);
    await dbWrapper.init();
    const db = dbWrapper.getAdapter();

    const cache = intializeDbCache(character, db);
    runtime = createAgent(character, db, cache, token) as AgentRuntime;

    runtime.actions = runtime.actions.filter((a) => a.name === "generate_image".toUpperCase())

    runtime.actions.map((a) => console.log(a.name + " " + a.description))

    const executeTask = async (task) => {
      console.log(`receieved a new task ${task.id} to the agent proposal ${task.proposalId} by user ${task.issuer}`)
      console.log(`task prompt: ${task.prompt}`)

      // TODO: Validate the task and propmpt

      // Execute the task
      runtime.character.topics = [task.prompt]
      // TODO: function shall return the tweet or url
      await runtime.clients.twitter.post.generateNewTweet()

      // Task Completion
      ensemble.completeTask(task.id, `Done tweet about topic: ${task.prompt}`)
    }

    // Adding the executeTask function as a listener so it will be called when a new task is received
    ensemble.setOnNewTaskListener(executeTask)

    try {
      await runtime.initialize();

    } catch (error) {
      elizaLogger.error("Failed to initialize runtime:", error);
      throw error;
    }

    const initializedClients = await initializeClients(character, runtime);
    clients = Object.values(initializedClients);
    runtime.clients = initializedClients;
    directClient.registerAgent(runtime);

    // ensemble.setOnNewTaskListener(async (task: TaskData) => {
    //   console.log('task', task);
    //   ensemble.sendProposal(task.id, '1');
    //   // runtime.processActions()
    //   // console.log('taskId', taskId);
    //   // console.log('runtime', runtime);
    //   // handleTask(taskId);
    // });
    // ensemble.setOnNewTaskListener((taskId) => {
    //   console.log('taskId', taskId);
    //   console.log('runtime', runtime);
    //   // handleTask(taskId);
    // });
    return clients;
  } catch (error) {
    elizaLogger.error(`Error starting agent for character ${character.name}:`, error);
    await cleanup(clients, runtime, dbWrapper);
    // Don't exit the process, just return null to indicate failure
    return null;
  }
}

// Update the cleanup function to use the database wrapper cleanup
async function cleanup(clients: any[], runtime: AgentRuntime | null, dbWrapper: DatabaseConnectionWrapper | null) {
  try {
    // Cleanup clients
    for (const client of clients) {
      try {
        await client?.cleanup?.();
      } catch (error) {
        elizaLogger.error('Error cleaning up client:', error);
      }
    }

    // Log runtime cleanup attempt
    if (runtime) {
      elizaLogger.log('Runtime cleanup skipped - no cleanup method available');
    }

    // Cleanup database connection
    if (dbWrapper) {
      await dbWrapper.cleanup();
    }
  } catch (error) {
    elizaLogger.error('Error during cleanup:', error);
  }
}

// Update the startAgents function to handle failures more gracefully
const startAgents = async () => {
  const directClient = new DirectClient();
  const serverPort = parseInt(settings.SERVER_PORT || "3000");
  const args = parseArguments();

  let charactersArg = args.characters || args.character;

  let characters = [defaultCharacter];

  if (charactersArg) {
      characters = await loadCharacters(charactersArg);
  }

  try {
      for (const character of characters) {
          await startAgent(character, directClient);
      }
  } catch (error) {
      elizaLogger.error("Error starting agents:", error);
  }

  // upload some agent functionality into directClient
  directClient.startAgent = async character => {
    // wrap it so we don't have to inject directClient later
    return startAgent(character, directClient)
  };
  directClient.start(serverPort);
  
};


startAgents().catch((error) => {
  elizaLogger.error("Unhandled error in startAgents:", error);
  process.exit(1);
});