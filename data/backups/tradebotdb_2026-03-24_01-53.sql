--
-- PostgreSQL database dump
--

\restrict bjbdujcgmsy3GOpVzlzHzF5l7U23ovYRxBx5ZUJE1VZdt8LGNMmFNE9vnTbjIVM

-- Dumped from database version 16.13 (Ubuntu 16.13-0ubuntu0.24.04.1)
-- Dumped by pg_dump version 16.13 (Ubuntu 16.13-0ubuntu0.24.04.1)

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

ALTER TABLE IF EXISTS ONLY public.users DROP CONSTRAINT IF EXISTS users_created_by_fkey;
ALTER TABLE IF EXISTS ONLY public.invitations DROP CONSTRAINT IF EXISTS invitations_used_by_fkey;
ALTER TABLE IF EXISTS ONLY public.invitations DROP CONSTRAINT IF EXISTS invitations_created_by_fkey;
ALTER TABLE IF EXISTS ONLY public.config_history DROP CONSTRAINT IF EXISTS config_history_user_id_fkey;
ALTER TABLE IF EXISTS ONLY public.audit_log DROP CONSTRAINT IF EXISTS audit_log_user_id_fkey;
DROP INDEX IF EXISTS public.ix_users_username;
DROP INDEX IF EXISTS public.ix_users_id;
DROP INDEX IF EXISTS public.ix_invitations_token;
DROP INDEX IF EXISTS public.ix_invitations_id;
DROP INDEX IF EXISTS public.ix_config_history_id;
DROP INDEX IF EXISTS public.ix_audit_log_id;
ALTER TABLE IF EXISTS ONLY public.users DROP CONSTRAINT IF EXISTS users_pkey;
ALTER TABLE IF EXISTS ONLY public.users DROP CONSTRAINT IF EXISTS users_email_key;
ALTER TABLE IF EXISTS ONLY public.invitations DROP CONSTRAINT IF EXISTS invitations_pkey;
ALTER TABLE IF EXISTS ONLY public.config_history DROP CONSTRAINT IF EXISTS config_history_pkey;
ALTER TABLE IF EXISTS ONLY public.audit_log DROP CONSTRAINT IF EXISTS audit_log_pkey;
ALTER TABLE IF EXISTS public.users ALTER COLUMN id DROP DEFAULT;
ALTER TABLE IF EXISTS public.invitations ALTER COLUMN id DROP DEFAULT;
ALTER TABLE IF EXISTS public.config_history ALTER COLUMN id DROP DEFAULT;
ALTER TABLE IF EXISTS public.audit_log ALTER COLUMN id DROP DEFAULT;
DROP SEQUENCE IF EXISTS public.users_id_seq;
DROP TABLE IF EXISTS public.users;
DROP SEQUENCE IF EXISTS public.invitations_id_seq;
DROP TABLE IF EXISTS public.invitations;
DROP SEQUENCE IF EXISTS public.config_history_id_seq;
DROP TABLE IF EXISTS public.config_history;
DROP SEQUENCE IF EXISTS public.audit_log_id_seq;
DROP TABLE IF EXISTS public.audit_log;
DROP TYPE IF EXISTS public.role_enum;
DROP TYPE IF EXISTS public.inv_role_enum;
--
-- Name: inv_role_enum; Type: TYPE; Schema: public; Owner: tradebot
--

CREATE TYPE public.inv_role_enum AS ENUM (
    'superadmin',
    'admin',
    'user'
);


ALTER TYPE public.inv_role_enum OWNER TO tradebot;

--
-- Name: role_enum; Type: TYPE; Schema: public; Owner: tradebot
--

CREATE TYPE public.role_enum AS ENUM (
    'superadmin',
    'admin',
    'user'
);


ALTER TYPE public.role_enum OWNER TO tradebot;

SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- Name: audit_log; Type: TABLE; Schema: public; Owner: tradebot
--

CREATE TABLE public.audit_log (
    id integer NOT NULL,
    user_id integer,
    username character varying(64),
    action character varying(128) NOT NULL,
    target character varying(256),
    details text,
    ip_address character varying(64),
    created_at timestamp without time zone
);


ALTER TABLE public.audit_log OWNER TO tradebot;

--
-- Name: audit_log_id_seq; Type: SEQUENCE; Schema: public; Owner: tradebot
--

CREATE SEQUENCE public.audit_log_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.audit_log_id_seq OWNER TO tradebot;

--
-- Name: audit_log_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: tradebot
--

ALTER SEQUENCE public.audit_log_id_seq OWNED BY public.audit_log.id;


--
-- Name: config_history; Type: TABLE; Schema: public; Owner: tradebot
--

CREATE TABLE public.config_history (
    id integer NOT NULL,
    user_id integer,
    username character varying(64),
    config_file character varying(128) NOT NULL,
    field_path character varying(256) NOT NULL,
    old_value text,
    new_value text,
    applied_at timestamp without time zone,
    rolled_back boolean
);


ALTER TABLE public.config_history OWNER TO tradebot;

--
-- Name: config_history_id_seq; Type: SEQUENCE; Schema: public; Owner: tradebot
--

CREATE SEQUENCE public.config_history_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.config_history_id_seq OWNER TO tradebot;

--
-- Name: config_history_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: tradebot
--

ALTER SEQUENCE public.config_history_id_seq OWNED BY public.config_history.id;


--
-- Name: invitations; Type: TABLE; Schema: public; Owner: tradebot
--

CREATE TABLE public.invitations (
    id integer NOT NULL,
    token character varying(128) NOT NULL,
    role public.inv_role_enum NOT NULL,
    email character varying(128),
    created_by integer,
    created_by_name character varying(64),
    expires_at timestamp without time zone NOT NULL,
    used_at timestamp without time zone,
    used_by integer
);


ALTER TABLE public.invitations OWNER TO tradebot;

--
-- Name: invitations_id_seq; Type: SEQUENCE; Schema: public; Owner: tradebot
--

CREATE SEQUENCE public.invitations_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.invitations_id_seq OWNER TO tradebot;

--
-- Name: invitations_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: tradebot
--

ALTER SEQUENCE public.invitations_id_seq OWNED BY public.invitations.id;


--
-- Name: users; Type: TABLE; Schema: public; Owner: tradebot
--

CREATE TABLE public.users (
    id integer NOT NULL,
    username character varying(64) NOT NULL,
    email character varying(128),
    password_hash character varying(256) NOT NULL,
    role public.role_enum NOT NULL,
    is_active boolean NOT NULL,
    created_by integer,
    created_at timestamp without time zone,
    last_login timestamp without time zone,
    failed_attempts integer,
    locked_until timestamp without time zone
);


ALTER TABLE public.users OWNER TO tradebot;

--
-- Name: users_id_seq; Type: SEQUENCE; Schema: public; Owner: tradebot
--

CREATE SEQUENCE public.users_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.users_id_seq OWNER TO tradebot;

--
-- Name: users_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: tradebot
--

ALTER SEQUENCE public.users_id_seq OWNED BY public.users.id;


--
-- Name: audit_log id; Type: DEFAULT; Schema: public; Owner: tradebot
--

ALTER TABLE ONLY public.audit_log ALTER COLUMN id SET DEFAULT nextval('public.audit_log_id_seq'::regclass);


--
-- Name: config_history id; Type: DEFAULT; Schema: public; Owner: tradebot
--

ALTER TABLE ONLY public.config_history ALTER COLUMN id SET DEFAULT nextval('public.config_history_id_seq'::regclass);


--
-- Name: invitations id; Type: DEFAULT; Schema: public; Owner: tradebot
--

ALTER TABLE ONLY public.invitations ALTER COLUMN id SET DEFAULT nextval('public.invitations_id_seq'::regclass);


--
-- Name: users id; Type: DEFAULT; Schema: public; Owner: tradebot
--

ALTER TABLE ONLY public.users ALTER COLUMN id SET DEFAULT nextval('public.users_id_seq'::regclass);


--
-- Data for Name: audit_log; Type: TABLE DATA; Schema: public; Owner: tradebot
--

COPY public.audit_log (id, user_id, username, action, target, details, ip_address, created_at) FROM stdin;
1	1	superadmin	LOGIN	superadmin	\N	127.0.0.1	2026-03-23 21:35:33.583065
2	1	superadmin	CREATE_USER	admin	role=user	127.0.0.1	2026-03-23 21:36:14.244903
3	1	superadmin	LOGOUT	\N	\N	127.0.0.1	2026-03-23 21:36:17.614176
4	2	admin	LOGIN	admin	\N	127.0.0.1	2026-03-23 21:36:32.861368
5	2	admin	LOGOUT	\N	\N	127.0.0.1	2026-03-23 21:39:57.330593
6	2	admin	LOGIN	admin	\N	127.0.0.1	2026-03-23 21:40:01.126079
7	2	admin	LOGOUT	\N	\N	127.0.0.1	2026-03-23 21:40:06.764444
8	1	superadmin	LOGIN	superadmin	\N	127.0.0.1	2026-03-23 21:40:10.017896
9	1	superadmin	CREATE_USER	admin1	role=admin	127.0.0.1	2026-03-23 21:40:34.347945
10	1	superadmin	LOGOUT	\N	\N	127.0.0.1	2026-03-23 21:40:37.372218
11	3	admin1	LOGIN	admin1	\N	127.0.0.1	2026-03-23 21:40:42.537351
12	3	admin1	TOGGLE_USER	admin	active=False	127.0.0.1	2026-03-23 21:40:59.736317
13	3	admin1	TOGGLE_USER	admin	active=True	127.0.0.1	2026-03-23 21:41:01.42503
\.


--
-- Data for Name: config_history; Type: TABLE DATA; Schema: public; Owner: tradebot
--

COPY public.config_history (id, user_id, username, config_file, field_path, old_value, new_value, applied_at, rolled_back) FROM stdin;
\.


--
-- Data for Name: invitations; Type: TABLE DATA; Schema: public; Owner: tradebot
--

COPY public.invitations (id, token, role, email, created_by, created_by_name, expires_at, used_at, used_by) FROM stdin;
\.


--
-- Data for Name: users; Type: TABLE DATA; Schema: public; Owner: tradebot
--

COPY public.users (id, username, email, password_hash, role, is_active, created_by, created_at, last_login, failed_attempts, locked_until) FROM stdin;
1	superadmin	\N	$2b$12$xPKyLqGJK4nFnFDCSQ6E6O/IC5dpaHKxfAemjDz0pJP5YfK44j34a	superadmin	t	\N	2026-03-23 21:32:04.456435	2026-03-23 21:40:09.951985	0	\N
2	admin	\N	$2b$12$vyh0mhz6GjGRGh9xLT0DJOsjbJcrh2YONvP1UUKDo1MsEEWJY9M8y	user	t	1	2026-03-23 21:36:14.231621	2026-03-23 21:40:01.110584	0	\N
3	admin1	\N	$2b$12$rvKJcn48Hpk./KRxFtSRKOvzKkUDKJpPuFgkuGNN/yadG6KBs8uaK	admin	t	1	2026-03-23 21:40:34.317383	2026-03-23 21:40:42.523929	0	\N
\.


--
-- Name: audit_log_id_seq; Type: SEQUENCE SET; Schema: public; Owner: tradebot
--

SELECT pg_catalog.setval('public.audit_log_id_seq', 1, false);


--
-- Name: config_history_id_seq; Type: SEQUENCE SET; Schema: public; Owner: tradebot
--

SELECT pg_catalog.setval('public.config_history_id_seq', 1, false);


--
-- Name: invitations_id_seq; Type: SEQUENCE SET; Schema: public; Owner: tradebot
--

SELECT pg_catalog.setval('public.invitations_id_seq', 1, false);


--
-- Name: users_id_seq; Type: SEQUENCE SET; Schema: public; Owner: tradebot
--

SELECT pg_catalog.setval('public.users_id_seq', 1, false);


--
-- Name: audit_log audit_log_pkey; Type: CONSTRAINT; Schema: public; Owner: tradebot
--

ALTER TABLE ONLY public.audit_log
    ADD CONSTRAINT audit_log_pkey PRIMARY KEY (id);


--
-- Name: config_history config_history_pkey; Type: CONSTRAINT; Schema: public; Owner: tradebot
--

ALTER TABLE ONLY public.config_history
    ADD CONSTRAINT config_history_pkey PRIMARY KEY (id);


--
-- Name: invitations invitations_pkey; Type: CONSTRAINT; Schema: public; Owner: tradebot
--

ALTER TABLE ONLY public.invitations
    ADD CONSTRAINT invitations_pkey PRIMARY KEY (id);


--
-- Name: users users_email_key; Type: CONSTRAINT; Schema: public; Owner: tradebot
--

ALTER TABLE ONLY public.users
    ADD CONSTRAINT users_email_key UNIQUE (email);


--
-- Name: users users_pkey; Type: CONSTRAINT; Schema: public; Owner: tradebot
--

ALTER TABLE ONLY public.users
    ADD CONSTRAINT users_pkey PRIMARY KEY (id);


--
-- Name: ix_audit_log_id; Type: INDEX; Schema: public; Owner: tradebot
--

CREATE INDEX ix_audit_log_id ON public.audit_log USING btree (id);


--
-- Name: ix_config_history_id; Type: INDEX; Schema: public; Owner: tradebot
--

CREATE INDEX ix_config_history_id ON public.config_history USING btree (id);


--
-- Name: ix_invitations_id; Type: INDEX; Schema: public; Owner: tradebot
--

CREATE INDEX ix_invitations_id ON public.invitations USING btree (id);


--
-- Name: ix_invitations_token; Type: INDEX; Schema: public; Owner: tradebot
--

CREATE UNIQUE INDEX ix_invitations_token ON public.invitations USING btree (token);


--
-- Name: ix_users_id; Type: INDEX; Schema: public; Owner: tradebot
--

CREATE INDEX ix_users_id ON public.users USING btree (id);


--
-- Name: ix_users_username; Type: INDEX; Schema: public; Owner: tradebot
--

CREATE UNIQUE INDEX ix_users_username ON public.users USING btree (username);


--
-- Name: audit_log audit_log_user_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: tradebot
--

ALTER TABLE ONLY public.audit_log
    ADD CONSTRAINT audit_log_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.users(id);


--
-- Name: config_history config_history_user_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: tradebot
--

ALTER TABLE ONLY public.config_history
    ADD CONSTRAINT config_history_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.users(id);


--
-- Name: invitations invitations_created_by_fkey; Type: FK CONSTRAINT; Schema: public; Owner: tradebot
--

ALTER TABLE ONLY public.invitations
    ADD CONSTRAINT invitations_created_by_fkey FOREIGN KEY (created_by) REFERENCES public.users(id);


--
-- Name: invitations invitations_used_by_fkey; Type: FK CONSTRAINT; Schema: public; Owner: tradebot
--

ALTER TABLE ONLY public.invitations
    ADD CONSTRAINT invitations_used_by_fkey FOREIGN KEY (used_by) REFERENCES public.users(id);


--
-- Name: users users_created_by_fkey; Type: FK CONSTRAINT; Schema: public; Owner: tradebot
--

ALTER TABLE ONLY public.users
    ADD CONSTRAINT users_created_by_fkey FOREIGN KEY (created_by) REFERENCES public.users(id);


--
-- PostgreSQL database dump complete
--

\unrestrict bjbdujcgmsy3GOpVzlzHzF5l7U23ovYRxBx5ZUJE1VZdt8LGNMmFNE9vnTbjIVM

